#include "dvd.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

using namespace aurora::dvd::impl;

namespace {

struct OverlayFileEntry {
  std::string fileName;
  void* userData;
  u32 size;
  s32 entryNum = k_invalidFstEntry;
  size_t sourceIndex = 0;
  u32 kind = AURORA_OVERLAY_FILE_NORMAL;
  s32 baseEntryNum = k_invalidFstEntry;
};

bool isBaseFile(s32 entryNum) {
  return entryNum > 0 && static_cast<size_t>(entryNum) < s_baseFileSizes.size() && s_baseFileSizes[entryNum] >= 0;
}

std::vector<OverlayFileEntry> s_overlayFiles;
std::unordered_map<std::string, s32> s_overlayEntryNums;
s32 s_nextOverlayEntryNum = 0;
s32 s_overlayEntryNumBase = 0;

std::string normalizeOverlayPath(std::string_view path) {
  std::string normalized;
  normalized.reserve(path.size());
  bool lastWasSlash = false;
  for (char ch : path) {
    if (ch == '\\') {
      ch = '/';
    }
    if (ch == '/') {
      if (lastWasSlash) {
        continue;
      }
      lastWasSlash = true;
      normalized.push_back('/');
      continue;
    }
    lastWasSlash = false;
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
    normalized.push_back(ch);
  }
  if (normalized.size() > 1 && normalized.back() == '/') {
    normalized.pop_back();
  }
  return normalized;
}

void syncOverlayEntryAllocator() {
  if (s_overlayEntryNumBase == s_baseEntryCount) {
    return;
  }

  s_overlayEntryNums.clear();
  s_overlayEntryNumBase = s_baseEntryCount;
  s_nextOverlayEntryNum = s_baseEntryCount;
}

s32 allocateOverlayEntryNum(std::string_view path) {
  syncOverlayEntryAllocator();

  std::string normalized = normalizeOverlayPath(path);
  auto it = s_overlayEntryNums.find(normalized);
  if (it != s_overlayEntryNums.end()) {
    return it->second;
  }

  const s32 entryNum = s_nextOverlayEntryNum++;
  s_overlayEntryNums.emplace(std::move(normalized), entryNum);
  return entryNum;
}


u32 fstCallback(u32 index, NodNodeKind kind, const char* name, u32 size, void* userData) {
  auto* ctx = static_cast<IterateContext*>(userData);

  while (index >= ctx->dirStack.back().second) {
    ctx->dirStack.pop_back();
  }

  const auto newEntry = std::make_shared<IterateNode>(
    name,
    (kind == NOD_NODE_KIND_DIRECTORY),
    size,
    index);

  if (s_baseFileSizes.size() <= index) {
    s_baseFileSizes.resize(index + 1, -1);
  }
  s_baseFileSizes[index] = newEntry->isDir ? -1 : static_cast<s64>(size);

  const auto& curDir = ctx->dirStack.back().first;
  curDir->children.push_back(newEntry);

  if (newEntry->isDir) {
    ctx->dirStack.emplace_back(newEntry, size);
  }

  return index + 1;
}

IterateNode* findNode(const IterateNode& node, const std::string_view name) {
  for (const auto& child : node.children) {
    if (nameEqualsIgnoreCase(child->name, name)) {
      return child.get();
    }
  }

  return nullptr;
}

void mergeOverlayFileIntoContext(const IterateContext& context, OverlayFileEntry& overlayFile) {
  IterateNode* node = context.root.get();
  std::string_view filePath = overlayFile.fileName;
  std::string currentPath;

  const bool isAlias = overlayFile.kind == AURORA_OVERLAY_FILE_BASE_ALIAS;
  if (isAlias && !isBaseFile(overlayFile.baseEntryNum)) {
    Log.error("Overlay file {} aliases entry {}, which is not a file on the disc!", overlayFile.fileName,
              overlayFile.baseEntryNum);
    return;
  }

  assert(filePath.starts_with('/'));
  filePath = filePath.substr(1);
  while (true) {
    const auto nextDelim = filePath.find('/');
    if (nextDelim == std::string_view::npos) {
      break;
    }

    const auto segment = filePath.substr(0, nextDelim);
    filePath = filePath.substr(nextDelim + 1);
    currentPath += '/';
    currentPath.append(segment);

    const auto existingNode = findNode(*node, segment);
    if (existingNode) {
      if (!existingNode->isDir) {
        Log.error("Overlay file {} needs directory that's already a file!", overlayFile.fileName);
        return;
      }

      node = existingNode;
    } else {
      const s32 entryNum = allocateOverlayEntryNum(currentPath);
      const auto newNode = std::make_shared<IterateNode>(std::string(segment), true, 0, entryNum);
      node->children.push_back(newNode);
      node = newNode.get();
    }
  }

  // Remainder of fileName is the actual file name, and node is the directory we're in.

  std::string fullFilePath = currentPath;
  fullFilePath += '/';
  fullFilePath.append(filePath);

  auto newNode = IterateNode(std::string(filePath), false, overlayFile.size, k_invalidFstEntry, overlayFile.userData);
  if (isAlias) {
    newNode = IterateNode(std::string(filePath), false, static_cast<u32>(s_baseFileSizes[overlayFile.baseEntryNum]),
                          k_invalidFstEntry);
    newNode.discEntryNum = overlayFile.baseEntryNum;
  }
  const auto existingNode = findNode(*node, filePath);
  if (existingNode) {
    if (existingNode->isDir) {
      Log.error("Overlay file {} overlaps directory with same name!", overlayFile.fileName);
      return;
    }

    newNode.originalEntryNum = existingNode->originalEntryNum;
    overlayFile.entryNum = newNode.originalEntryNum;

    // Replace existing disc entry.
    *existingNode = std::move(newNode);
  } else {
    // Add new entry.
    newNode.originalEntryNum = allocateOverlayEntryNum(fullFilePath);
    overlayFile.entryNum = newNode.originalEntryNum;

    node->children.emplace_back(std::make_shared<IterateNode>(std::move(newNode)));
  }
}

void mergeOverlayFilesIntoContext(const IterateContext& context) {
  for (auto& overlayFile : s_overlayFiles) {
    mergeOverlayFileIntoContext(context, overlayFile);
  }
}

void makeFstRecursive(IterateNode& node, FstIndex parent) {
  if (node.originalEntryNum != k_invalidFstEntry) {
    if (s_entryNumToFstIndex.size() <= node.originalEntryNum) {
      s_entryNumToFstIndex.resize(node.originalEntryNum + 1, k_invalidFstEntry);
    }

    auto& map = s_entryNumToFstIndex[node.originalEntryNum];
    if (map != k_invalidFstEntry) {
      Log.error("File {} with entry num {} already exists in map!", node.name, node.originalEntryNum);
      return;
    }

    map = static_cast<FstIndex>(s_fstEntries.size());
  }

  if (!node.isDir) {
    assert(node.children.empty());
    assert(node.originalEntryNum != k_invalidFstEntry);

    s_fstEntries.emplace_back(node.name, false, parent, node.size, node.overlayData, node.isOverlay,
                              node.originalEntryNum, node.discEntryNum);
    return;
  }

  std::ranges::sort(node.children, [](const auto& a, const auto& b) { return a->name < b->name; });

  const FstIndex ourIndex = static_cast<FstIndex>(s_fstEntries.size());
  s_fstEntries.emplace_back(node.name, true, parent, 0, node.overlayData, node.isOverlay, node.originalEntryNum,
                            node.discEntryNum);

  for (const auto& child : node.children) {
    makeFstRecursive(*child, ourIndex);
  }

  s_fstEntries[ourIndex].nextOrLength = static_cast<u32>(s_fstEntries.size());
}

void makeFstFromContext(const IterateContext& context) {
  makeFstRecursive(*context.root, 0);
}

s32 calcEntryCount(const IterateNode& node) {
  s32 counter = 1;

  for (const auto& child : node.children) {
    counter += calcEntryCount(*child);
  }

  return counter;
}

bool validateOverlayFile(const AuroraOverlayFile& file) {
  const std::string_view name(file.fileName);

  if (!name.starts_with('/')) {
    Log.error("Overlay path {} does not start with /", name);
    return false;
  }

  if (file.size > std::numeric_limits<u32>::max()) {
    Log.error("Overlay file sizes above 4 GiB are not supported: {}", name);
    return false;
  }

  if (file.kind != AURORA_OVERLAY_FILE_NORMAL && file.kind != AURORA_OVERLAY_FILE_BASE_ALIAS) {
    Log.error("Overlay file {} has unknown kind {}", name, file.kind);
    return false;
  }

  return true;
}

}

namespace aurora::dvd::impl {

bool rebuildFST() {
  using namespace std::string_literals;

  if (s_partition == nullptr) {
    return false;
  }

  std::lock_guard lock(s_fstLock);

  s32 currentDirEntryNum = k_invalidFstEntry;
  const std::string currentPath = s_currentPath;
  if (s_currentDir >= 0 && static_cast<size_t>(s_currentDir) < s_fstEntries.size() && s_fstEntries[s_currentDir].isDir) {
    currentDirEntryNum = s_fstEntries[s_currentDir].origEntryNum;
  }

  s_fstEntries.clear();
  s_entryNumToFstIndex.clear();
  s_baseFileSizes.assign(1, -1);
  IterateContext ctx;
  ctx.root = std::make_shared<IterateNode>(""s, true, 0, 0);
  ctx.dirStack.emplace_back(ctx.root, std::numeric_limits<u32>::max());

  nod_partition_iterate_fst(s_partition, fstCallback, &ctx);
  s_baseEntryCount = calcEntryCount(*ctx.root);
  syncOverlayEntryAllocator();
  mergeOverlayFilesIntoContext(ctx);
  makeFstFromContext(ctx);

  if (currentDirEntryNum >= 0 && static_cast<size_t>(currentDirEntryNum) < s_entryNumToFstIndex.size()) {
    const FstIndex currentDir = s_entryNumToFstIndex[currentDirEntryNum];
    if (currentDir >= 0 && static_cast<size_t>(currentDir) < s_fstEntries.size() && s_fstEntries[currentDir].isDir) {
      s_currentDir = currentDir;
      s_currentPath = currentPath;
      return true;
    }
  }

  if (currentDirEntryNum != k_invalidFstEntry) {
    Log.warn("Current DVD directory {} with entryNum {} was lost during FST rebuild; resetting to root",
             currentPath, currentDirEntryNum);
  }

  s_currentDir = 0;
  s_currentPath = "/";
  return true;
}

bool nameEqualsIgnoreCase(const std::string_view lhs, const std::string_view rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < rhs.size(); ++i) {
    char lc = lhs[i];
    char rc = rhs[i];
    if (lc >= 'a' && lc <= 'z') {
      lc = static_cast<char>(lc - 'a' + 'A');
    }
    if (rc >= 'a' && rc <= 'z') {
      rc = static_cast<char>(rc - 'a' + 'A');
    }
    if (lc != rc) {
      return false;
    }
  }
  return true;
}

}

s32 aurora_dvd_base_entry_count() {
  return s_baseEntryCount;
}

s32 aurora_dvd_base_entry_num(const char* path) {
  if (path == nullptr || path[0] != '/') {
    return k_invalidFstEntry;
  }
  // Disc files keep their EntryNum under overlays, so the current tree finds them.
  const s32 entryNum = DVDConvertPathToEntrynum(path);
  std::lock_guard lock(s_fstLock);
  return isBaseFile(entryNum) ? entryNum : k_invalidFstEntry;
}

int64_t aurora_dvd_base_file_size(s32 baseEntryNum) {
  std::lock_guard lock(s_fstLock);
  return isBaseFile(baseEntryNum) ? s_baseFileSizes[baseEntryNum] : -1;
}

int64_t aurora_dvd_read_base_file(s32 baseEntryNum, void* buf, size_t len, uint64_t offset) {
  if (buf == nullptr && len != 0) {
    return -1;
  }

  NodHandle* handle = nullptr;
  {
    std::lock_guard lock(s_fstLock);
    if (s_partition == nullptr || !isBaseFile(baseEntryNum) ||
        nod_partition_open_file(s_partition, baseEntryNum, &handle) != NOD_RESULT_OK || handle == nullptr) {
      return -1;
    }
  }

  int64_t total = 0;
  if (offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      nod_seek(handle, static_cast<int64_t>(offset), 0) < 0) {
    total = -1;
  }
  auto* out = static_cast<uint8_t*>(buf);
  while (total >= 0 && static_cast<size_t>(total) < len) {
    const int64_t read = nod_read(handle, out + total, len - static_cast<size_t>(total));
    if (read < 0) {
      total = -1;
    } else if (read == 0) {
      break;
    } else {
      total += read;
    }
  }
  nod_free(handle);
  return total;
}

void aurora_dvd_overlay_files(const AuroraOverlayFile* files, size_t nFiles, s32* outEntryNums) {
  if (!s_overlayCallbacksSet) {
    Log.fatal("aurora_dvd_overlay_callbacks not called before aurora_dvd_overlay_files!");
  }

  s_overlayFiles.clear();
  if (outEntryNums != nullptr) {
    std::fill_n(outEntryNums, nFiles, k_invalidFstEntry);
  }

  for (size_t i = 0; i < nFiles; i++) {
    const auto& file = files[i];

    if (!validateOverlayFile(file)) {
      continue;
    }

    s_overlayFiles.emplace_back(file.fileName, file.userData, static_cast<u32>(file.size), k_invalidFstEntry, i,
                                file.kind, file.baseEntryNum);
  }

  rebuildFST();

  if (outEntryNums != nullptr) {
    for (const auto& file : s_overlayFiles) {
      if (file.entryNum != k_invalidFstEntry) {
        outEntryNums[file.sourceIndex] = file.entryNum;
      }
    }
  }
}

void aurora_dvd_overlay_callbacks(const AuroraOverlayCallbacks* callbacks) {
  s_overlayCallbacks = *callbacks;
  s_overlayCallbacksSet = true;
}
