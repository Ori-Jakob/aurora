#pragma once

#include <string>
#include <memory>
#include <mutex>

#include <nod.h>

#include <dolphin/os.h>
#include <aurora/dvd.h>
#include <dolphin/dvd.h>

#include "../../internal.hpp"

namespace aurora::dvd::impl {

inline Module Log("aurora::dvd");

using FstIndex = s32;
constexpr s32 k_invalidFstEntry = -1;

struct FSTEntry {
  std::string name;
  bool isDir = false;
  FstIndex parent = 0;
  u32 nextOrLength = 0;
  void* overlayData = nullptr;
  bool isOverlay = false;
  s32 origEntryNum = 0;
  // Base disc file a non-overlay file reads from: origEntryNum, or the source of a base alias.
  s32 discEntryNum = 0;
};

struct IterateNode {
  std::string name;
  bool isDir;
  s32 originalEntryNum;
  u32 size;
  void* overlayData;
  bool isOverlay;
  s32 discEntryNum;
  std::vector<std::shared_ptr<IterateNode>> children;

  IterateNode(std::string name, bool isDir, u32 size, s32 originalEntryNum, void* overlayData)
  : name(std::move(name)), isDir(isDir), size(size), originalEntryNum(originalEntryNum), overlayData(overlayData), isOverlay(true), discEntryNum(originalEntryNum) {}

  IterateNode(std::string name, bool isDir, u32 size, s32 originalEntryNum)
  : name(std::move(name)), isDir(isDir), size(size), originalEntryNum(originalEntryNum), overlayData(nullptr), isOverlay(false), discEntryNum(originalEntryNum) {}
};

struct IterateContext {
  std::shared_ptr<IterateNode> root;
  std::vector<std::pair<std::shared_ptr<IterateNode>, u32>> dirStack;
};

extern NodHandle* s_partition;
extern std::vector<FSTEntry> s_fstEntries;
// Map from public FST entryNums (matching base disc, Aurora-assigned for new overlay files)
// To the current FST indexes (that we use for navigating the tree).
// Unfilled spots are given the k_invalidFstEntry value.
extern std::vector<FstIndex> s_entryNumToFstIndex;
extern s32 s_baseEntryCount;
// Size of each base disc entry by EntryNum, -1 for directories.
extern std::vector<s64> s_baseFileSizes;
extern FstIndex s_currentDir;
extern std::string s_currentPath;
extern BOOL s_autoInvalidation;
extern BOOL s_autoFatalMessaging;
extern DVDDiskID s_diskID;
extern DVDLowCallback s_resetCoverCallback;
extern bool s_initialized;
extern bool s_overlayCallbacksSet;
extern AuroraOverlayCallbacks s_overlayCallbacks;
extern std::mutex s_fstLock;

bool rebuildFST();
bool nameEqualsIgnoreCase(std::string_view lhs, std::string_view rhs);

}
