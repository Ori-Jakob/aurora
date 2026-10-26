#include <cstring>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <filesystem>
#include <utility>
#include <vector>

#include "gpu.hpp"
#include "../internal.hpp"
#include "../io.hpp"
#include "../sqlite_utils.hpp"

#include <sqlite3.h>
#include <fmt/format.h>
#if defined(AURORA_CACHE_USE_ZSTD)
#include <zstd.h>
#endif
#define XXH_STATIC_LINKING_ONLY
#include <xxhash.h>

namespace aurora::webgpu {
static Module Log("aurora::gpu::cache");

static sqlite3* db;
static sqlite3_stmt* load_stmt;
static sqlite3_stmt* store_stmt;
static sqlite3_stmt* tag_stmt;
static bool cache_broken;
static std::mutex cache_mutex;
static std::vector<XXH128_hash_t> cache_keys_used;

struct LayoutTag {
  XXH128_hash_t key;
  uint64_t layout;
};
static std::vector<LayoutTag> pending_layout_tags;
static thread_local uint64_t t_cacheLayout = 0;
static int64_t cache_session;
#if defined(AURORA_CACHE_USE_ZSTD)
static std::vector<uint8_t> compress_buffer;
#endif

constexpr int CACHE_SCHEMA = 2;
// % of rows pruned to trigger a full VACUUM
constexpr uint64_t VacuumPrunePercentThreshold = 25;

static std::filesystem::path cache_path() {
  return io::fs_path_from_string(g_config.cachePath) / "dawn_cache.db";
}

static void init_abort() {
  cache_broken = true;
  sqlite3_close(db);
  db = nullptr;
}

static int check(int ret) {
  if (ret != SQLITE_OK) {
    Log.error("SQLite operation failed: {}", sqlite3_errmsg(db));
  }

  return ret;
}

static bool ensure_schema_up_to_date() {
  sqlite::Transaction tx(db, Log, true);
  if (!tx) {
    Log.error("Failed to open schema check transaction", sqlite3_errmsg(db));
    return false;
  }

  auto ret = sqlite::exec(db, "CREATE TABLE IF NOT EXISTS aurora_schema(value INTEGER);");
  if (ret != SQLITE_OK) {
    Log.error("Failed to create schema table: {}", sqlite3_errmsg(db));
    return false;
  }

  bool match = false;
  auto cmd = fmt::format("SELECT * FROM aurora_schema WHERE value = {}", CACHE_SCHEMA);
  ret = sqlite::exec(db, cmd.c_str(), [&match](int, char**, char**) { match = true; }, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to check schema table: {}", sqlite3_errmsg(db));
    return false;
  }

  if (match) {
    return true;
  }

  cmd = fmt::format(
      R"(DROP TABLE IF EXISTS cache;
DROP TABLE IF EXISTS cache_layout_tags;
CREATE TABLE cache (
  key BLOB PRIMARY KEY NOT NULL,
  value BLOB NOT NULL,
  size INTEGER NOT NULL,
  compressed INTEGER NOT NULL
);
DELETE FROM aurora_schema;
INSERT INTO aurora_schema VALUES ({});)",
      CACHE_SCHEMA);
  ret = sqlite::exec(db, cmd.c_str());
  if (ret != SQLITE_OK) {
    Log.error("Failed to update schema: {}", sqlite3_errmsg(db));
    return false;
  }

  tx.commit();
  return true;
}

static std::optional<uint64_t> select_uint64(const char* sql) {
  sqlite3_stmt* stmt = nullptr;
  auto ret = sqlite3_prepare_v3(db, sql, -1, 0, &stmt, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare statement '{}': {}", sql, sqlite3_errmsg(db));
    return std::nullopt;
  }

  std::optional<uint64_t> result;
  ret = sqlite3_step(stmt);
  if (ret == SQLITE_ROW) {
    result = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
  } else if (ret != SQLITE_DONE) {
    Log.error("Failed to execute statement '{}': {}", sql, sqlite3_errmsg(db));
  }

  sqlite3_finalize(stmt);
  return result;
}

static bool cache_init_core() {
  Log.debug("SQLite version {}", sqlite3_libversion());

  std::string file = io::fs_path_to_string(cache_path());
  Log.debug("Using dawn cache at {}", file);
  auto ret = sqlite3_open(file.c_str(), &db);
  if (ret != SQLITE_OK) {
    Log.error("Failed to open database: {}", sqlite3_errmsg(db));
    return false;
  }

  // WAL mode + NORMAL = no need for disk syncs, consistent but not durable is fine.
  ret = sqlite::exec(db, "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
  if (ret != SQLITE_OK) {
    Log.error("Failed to set pragmas: {}", sqlite3_errmsg(db));
    return false;
  }

  if (!ensure_schema_up_to_date()) {
    Log.error("Failed to validate schema");
    return false;
  }

  // Kept apart from the cache table so adding it leaves existing blobs valid
  ret = sqlite::exec(db,
                     "CREATE TABLE IF NOT EXISTS cache_layout_tags ("
                     "key BLOB NOT NULL,"
                     "layout INTEGER NOT NULL,"
                     "session INTEGER NOT NULL,"
                     "PRIMARY KEY (key, layout)"
                     ") WITHOUT ROWID;");
  if (ret != SQLITE_OK) {
    Log.error("Failed to create layout table: {}", sqlite3_errmsg(db));
    return false;
  }

  // Sessions count only runs that tagged a blob, so a layout's age is the runs it sat idle through
  const auto lastSession = select_uint64("SELECT COALESCE(MAX(session), 0) FROM cache_layout_tags");
  if (!lastSession) {
    return false;
  }
  cache_session = static_cast<int64_t>(*lastSession) + 1;

  ret = sqlite3_prepare_v3(db, "SELECT value, size, compressed FROM cache WHERE key = ?", -1, SQLITE_PREPARE_PERSISTENT,
                           &load_stmt, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare statement: {}", sqlite3_errmsg(db));
    return false;
  }

  ret = sqlite3_prepare_v3(db, "REPLACE INTO cache (key, value, size, compressed) VALUES (?, ?, ?, ?)", -1,
                           SQLITE_PREPARE_PERSISTENT, &store_stmt, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare statement: {}", sqlite3_errmsg(db));
    return false;
  }

  ret = sqlite3_prepare_v3(db, "REPLACE INTO cache_layout_tags (key, layout, session) VALUES (?, ?, ?)", -1,
                           SQLITE_PREPARE_PERSISTENT, &tag_stmt, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare statement: {}", sqlite3_errmsg(db));
    return false;
  }

  return true;
}

static bool fill_used_key_table(const std::vector<XXH128_hash_t>& usedKeys) {
  auto ret = sqlite::exec(db,
                          "CREATE TEMP TABLE IF NOT EXISTS cache_keys_used ("
                          "key BLOB PRIMARY KEY NOT NULL"
                          ");"
                          "DELETE FROM cache_keys_used;");
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare dawn cache prune key table: {}", sqlite3_errmsg(db));
    return false;
  }

  sqlite3_stmt* stmt = nullptr;
  ret = sqlite3_prepare_v3(db, "INSERT OR IGNORE INTO cache_keys_used (key) VALUES (?)", -1, 0, &stmt, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare dawn cache prune key insert: {}", sqlite3_errmsg(db));
    return false;
  }

  for (const auto& keyHash : usedKeys) {
    ret = sqlite3_bind_blob(stmt, 1, &keyHash, sizeof(keyHash), SQLITE_TRANSIENT);
    if (ret != SQLITE_OK) {
      Log.error("Failed to bind dawn cache prune key: {}", sqlite3_errmsg(db));
      sqlite3_finalize(stmt);
      return false;
    }

    ret = sqlite3_step(stmt);
    if (ret != SQLITE_DONE) {
      Log.error("Failed to insert dawn cache prune key: {}", sqlite3_errmsg(db));
      sqlite3_finalize(stmt);
      return false;
    }

    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);
  }

  sqlite3_finalize(stmt);
  return true;
}

static void note_layout(const XXH128_hash_t& keyHash) {
  if (t_cacheLayout != 0) {
    pending_layout_tags.push_back({keyHash, t_cacheLayout});
  }
}

// Loads stay read-only, so their tags wait for the next write transaction
static bool flush_layout_tags() {
  for (const auto& tag : pending_layout_tags) {
    check(sqlite3_bind_blob(tag_stmt, 1, &tag.key, sizeof(tag.key), SQLITE_TRANSIENT));
    check(sqlite3_bind_int64(tag_stmt, 2, static_cast<sqlite3_int64>(tag.layout)));
    check(sqlite3_bind_int64(tag_stmt, 3, cache_session));
    const auto ret = sqlite3_step(tag_stmt);
    sqlite3_reset(tag_stmt);
    if (ret != SQLITE_DONE) {
      Log.error("Failed to record dawn cache layout: {}", sqlite3_errmsg(db));
      return false;
    }
  }
  pending_layout_tags.clear();
  return true;
}

static bool fill_known_layout_table(std::span<const uint64_t> knownLayouts, std::span<const uint64_t> liveLayouts) {
  auto ret = sqlite::exec(db,
                          "CREATE TEMP TABLE IF NOT EXISTS cache_layouts_known ("
                          "layout INTEGER PRIMARY KEY NOT NULL,"
                          "live INTEGER NOT NULL"
                          ");"
                          "DELETE FROM cache_layouts_known;");
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare dawn cache known layout table: {}", sqlite3_errmsg(db));
    return false;
  }

  sqlite3_stmt* stmt = nullptr;
  ret = sqlite3_prepare_v3(db, "INSERT OR IGNORE INTO cache_layouts_known (layout, live) VALUES (?, ?)", -1, 0, &stmt,
                           nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare dawn cache known layout insert: {}", sqlite3_errmsg(db));
    return false;
  }

  // Live rows go first so a layout in both lists stays live
  for (const auto& [layouts, live] : {std::pair{liveLayouts, 1}, std::pair{knownLayouts, 0}}) {
    for (const uint64_t layout : layouts) {
      check(sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(layout)));
      check(sqlite3_bind_int(stmt, 2, live));
      ret = sqlite3_step(stmt);
      sqlite3_reset(stmt);
      if (ret != SQLITE_DONE) {
        Log.error("Failed to insert dawn cache known layout: {}", sqlite3_errmsg(db));
        sqlite3_finalize(stmt);
        return false;
      }
    }
  }

  sqlite3_finalize(stmt);
  return true;
}

static bool cache_init() {
  if (cache_broken) {
    return false;
  }

  if (db) {
    return true;
  }

  if (!cache_init_core()) {
    Log.error("SQLite DB init failed");
    init_abort();
    return false;
  }

  Log.debug("SQLite cache init succeeded");

  return true;
}

size_t load_from_cache(void const* key, size_t keySize, void* value, size_t valueSize, void*) {
  std::lock_guard lock(cache_mutex);

  if (!cache_init()) {
    return 0;
  }

  sqlite::Transaction tx(db, Log);
  if (!tx) {
    Log.error("Failed to open load transaction");
    return 0;
  }

  const auto keyHash = XXH128(key, keySize, 0);
  check(sqlite3_bind_blob(load_stmt, 1, &keyHash, sizeof(keyHash), SQLITE_TRANSIENT));

  const auto ret = sqlite3_step(load_stmt);
  size_t foundSize;
  bool loadSucceeded = false;
  if (ret == SQLITE_ROW) {
    // Hit
    const auto foundPtr = sqlite3_column_blob(load_stmt, 0);
    foundSize = sqlite3_column_int64(load_stmt, 1);
    const bool compressed = sqlite3_column_int(load_stmt, 2) != 0;

    if (value && valueSize == foundSize) {
      loadSucceeded = true;
      if (compressed) {
#if defined(AURORA_CACHE_USE_ZSTD)
        const auto compSize = sqlite3_column_bytes(load_stmt, 0);
        const auto zstdRet = ZSTD_decompress(value, valueSize, foundPtr, compSize);
        if (ZSTD_isError(zstdRet)) {
          Log.error("zstd decompression error: {}", ZSTD_getErrorName(zstdRet));
          foundSize = 0;
          loadSucceeded = false;
        } else if (zstdRet != foundSize) {
          Log.error("zstd decompression size mismatch: expected {}, got {}", foundSize, zstdRet);
          foundSize = 0;
          loadSucceeded = false;
        }
#else
        Log.error("Cache entry is zstd-compressed but zstd support is disabled");
        foundSize = 0;
        loadSucceeded = false;
#endif
      } else {
        if (foundSize != 0 && !foundPtr) {
          Log.error("Cache entry is missing raw value data");
          foundSize = 0;
          loadSucceeded = false;
        } else if (foundSize != 0) {
          std::memcpy(value, foundPtr, foundSize);
        }
      }
    }
  } else if (ret == SQLITE_DONE) {
    // Miss
    foundSize = 0;
  } else {
    Log.error("Looking up cache key failed: {}", sqlite3_errmsg(db));
    return 0;
  }

  check(sqlite3_reset(load_stmt));

  if (loadSucceeded) {
    cache_keys_used.push_back(keyHash);
    note_layout(keyHash);
  }

  return foundSize;
}

void store_to_cache(void const* key, size_t keySize, void const* value, size_t valueSize, void*) {
  std::lock_guard lock(cache_mutex);

  if (!cache_init()) {
    return;
  }

  sqlite::Transaction tx(db, Log, true);
  if (!tx) {
    Log.error("Failed to open store transaction");
    return;
  }

  const void* storedValue = value;
  sqlite3_uint64 storedValueSize = valueSize;
  int compressed = 0;
#if defined(AURORA_CACHE_USE_ZSTD)
  const auto bound = ZSTD_compressBound(valueSize);
  if (ZSTD_isError(bound)) {
    Log.error("Failed to calculate ZSTD_compressBound: {}", ZSTD_getErrorName(bound));
    return;
  }

  if (compress_buffer.size() < bound) {
    compress_buffer.resize(bound);
  }

  const auto compressRet = ZSTD_compress(compress_buffer.data(), compress_buffer.size(), value, valueSize, 0);
  if (ZSTD_isError(compressRet)) {
    Log.error("ZSTD compression error: {}", ZSTD_getErrorName(compressRet));
    return;
  }

  if (compressRet < valueSize) {
    storedValue = compress_buffer.data();
    storedValueSize = compressRet;
    compressed = 1;
  }
#endif

  const auto keyHash = XXH128(key, keySize, 0);
  check(sqlite3_bind_blob64(store_stmt, 1, &keyHash, sizeof(keyHash), SQLITE_TRANSIENT));
  check(
      sqlite3_bind_blob64(store_stmt, 2, storedValue, storedValueSize, compressed ? SQLITE_STATIC : SQLITE_TRANSIENT));
  check(sqlite3_bind_int64(store_stmt, 3, static_cast<sqlite3_int64>(valueSize)));
  check(sqlite3_bind_int(store_stmt, 4, compressed));

  const auto ret = sqlite3_step(store_stmt);
  if (ret != SQLITE_DONE) {
    // Error or something
    Log.error("Failed to insert row: {}", sqlite3_errmsg(db));
    return;
  }

  check(sqlite3_reset(store_stmt));
  check(sqlite3_bind_null(store_stmt, 2));
  check(sqlite3_bind_null(store_stmt, 4));

  note_layout(keyHash);
  flush_layout_tags();

  tx.commit();
  cache_keys_used.push_back(keyHash);
}

void cache_prune(std::span<const uint64_t> knownLayouts, std::span<const uint64_t> liveLayouts) {
  std::lock_guard lock(cache_mutex);
  if (!cache_init()) {
    return;
  }
  if (cache_keys_used.empty()) {
    Log.warn("Skipping Dawn cache prune because no cache keys were used");
    return;
  }

  uint64_t totalRows = 0;
  sqlite3_int64 deletedRows = 0;
  {
    sqlite::Transaction tx(db, Log, true);
    if (!tx) {
      Log.error("Failed to open dawn cache prune transaction");
      return;
    }

    if (!flush_layout_tags() || !fill_used_key_table(cache_keys_used) ||
        !fill_known_layout_table(knownLayouts, liveLayouts)) {
      return;
    }

    const auto totalRowsResult = select_uint64("SELECT COUNT(*) FROM cache");
    if (!totalRowsResult) {
      return;
    }
    totalRows = *totalRowsResult;

    // A live layout loads every reachable blob before the prune, so only an idle known layout keeps unused ones
    const int64_t lapsedSession = cache_session - CacheLayoutIdleSessions;
    auto sql = fmt::format(
        "DELETE FROM cache "
        "WHERE NOT EXISTS ("
        "  SELECT 1 FROM cache_keys_used WHERE cache_keys_used.key = cache.key"
        ") AND NOT EXISTS ("
        "  SELECT 1 FROM cache_layout_tags JOIN cache_layouts_known USING (layout)"
        "  WHERE cache_layout_tags.key = cache.key AND live = 0 AND session > {}"
        ")",
        lapsedSession);
    auto ret = sqlite::exec(db, sql.c_str());
    if (ret != SQLITE_OK) {
      Log.error("Failed to prune dawn cache rows: {}", sqlite3_errmsg(db));
      return;
    }
    deletedRows = sqlite3_changes64(db);

    sql = fmt::format(
        "DELETE FROM cache_layout_tags "
        "WHERE session <= {} OR layout NOT IN (SELECT layout FROM cache_layouts_known) OR NOT EXISTS ("
        "  SELECT 1 FROM cache WHERE cache.key = cache_layout_tags.key"
        ")",
        lapsedSession);
    ret = sqlite::exec(db, sql.c_str());
    if (ret != SQLITE_OK) {
      Log.error("Failed to prune dawn cache layouts: {}", sqlite3_errmsg(db));
      return;
    }

    tx.commit();
  }

  if (deletedRows == 0) {
    Log.debug("Dawn cache prune completed; no stale entries found");
    return;
  }

  Log.info("Pruned {} stale Dawn cache entries", deletedRows);

  // VACUUM if we removed at least 25% of the rows
  if (totalRows != 0 && static_cast<uint64_t>(deletedRows) * 100ull >= totalRows * VacuumPrunePercentThreshold) {
    if (const auto ret = sqlite::exec(db, "VACUUM;"); ret != SQLITE_OK) {
      Log.warn("Failed to vacuum dawn cache after pruning: {}", sqlite3_errmsg(db));
      return;
    }
  }

  if (const auto ret = sqlite::exec(db, "PRAGMA wal_checkpoint(TRUNCATE);"); ret != SQLITE_OK) {
    Log.warn("Failed to checkpoint dawn cache WAL: {}", sqlite3_errmsg(db));
  }
}

void cache_shutdown() {
#if defined(AURORA_CACHE_USE_ZSTD)
  compress_buffer.clear();
#endif
  cache_keys_used.clear();
  if (db != nullptr && tag_stmt != nullptr && !pending_layout_tags.empty()) {
    sqlite::Transaction tx(db, Log, true);
    if (tx && flush_layout_tags()) {
      tx.commit();
    }
  }
  pending_layout_tags.clear();
  if (tag_stmt != nullptr) {
    check(sqlite3_finalize(tag_stmt));
    tag_stmt = nullptr;
  }
  if (load_stmt != nullptr) {
    check(sqlite3_finalize(load_stmt));
    load_stmt = nullptr;
  }
  if (store_stmt != nullptr) {
    check(sqlite3_finalize(store_stmt));
    store_stmt = nullptr;
  }
  if (db != nullptr) {
    check(sqlite3_close(db));
    db = nullptr;
  }
}

CacheLayoutScope::CacheLayoutScope(uint64_t layoutKey) noexcept : m_previous(std::exchange(t_cacheLayout, layoutKey)) {}

CacheLayoutScope::~CacheLayoutScope() { t_cacheLayout = m_previous; }

} // namespace aurora::webgpu
