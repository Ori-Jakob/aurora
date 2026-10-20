#ifndef AURORA_DVD_H
#define AURORA_DVD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <dolphin/types.h>

/**
 * Open a GC/Wii disc image for use by the DVD API.
 * Must be called before DVDInit().
 * Returns true on success, false on failure.
 */
bool aurora_dvd_open(const char* disc_path);

/**
 * Close the disc image and free all resources.
 */
void aurora_dvd_close(void);

/**
 * OVERLAY FILES!
 *
 * Overlay files allow you to replace and add ("overlay") files that are present in the loaded DVD.
 * The way this works is pretty simple: you provide some callbacks and a list of files.
 * When an overlaid file gets read, your callbacks get called instead of pulling from the underlying DVD.
 *
 * Original disc EntryNums are not touched by the overlay system. New files and directories
 * are assigned stable EntryNums by Aurora.
 */

/**
 * \brief How an overlay file is served.
 */
typedef enum AuroraOverlayFileKind {
  /**
   * \brief Served through the overlay callbacks.
   */
  AURORA_OVERLAY_FILE_NORMAL = 0,

  /**
   * \brief Served from the base disc file baseEntryNum, under a path of its own.
   *
   * userData is unused and the size is taken from the base disc file.
   */
  AURORA_OVERLAY_FILE_BASE_ALIAS = 1,
} AuroraOverlayFileKind;

/**
 * \brief A single file to be overlaid over the DVD files.
 *
 * You do not need to concern yourself with providing entries for directories. They are automatically merged
 * and created where necessary.
 *
 * Zero-initialized fields after size keep the plain callback behavior.
 */
typedef struct AuroraOverlayFile {
  /**
   * \brief Absolute file path of this file.
   *
   * Must be in the form "/foo/bar/baz.txt", note the leading slash.
   */
  const char* fileName;

  /**
   * \brief Userdata pointer that will be passed to the callback when this file is opened.
   */
  void* userData;

  /**
   * \brief Size of this file, in bytes.
   *
   * While this is of type size_t, file sizes larger than u32 are not currently supported.
   */
  size_t size;

  /**
   * \brief An AuroraOverlayFileKind.
   */
  uint32_t kind;

  /**
   * \brief For AURORA_OVERLAY_FILE_BASE_ALIAS: EntryNum of the base disc file to serve (see aurora_dvd_base_entry_num).
   */
  int32_t baseEntryNum;

} AuroraOverlayFile;

/**
 * \brief Callbacks to implement overlay files.
 *
 * Callbacks may be ran from any thread at any time. Make sure they're thread safe!
 */
typedef struct AuroraOverlayCallbacks {
  /**
   * Called when a new file has been opened.
   *
   * Returns an opaque handle that will be passed to the remaining callbacks. Receives the userdata specified in
   * the AuroraOverlayFile.
   */
  void* (*open)(void* userdata);

  /**
   * Close a file handle previously returned from the open callback.
   */
  void (*close)(void* handle);

  /**
   * Read data from a file handle.
   *
   * Returns the amount of data read, or -1 on error.
   */
  int64_t (*read)(void* handle, uint8_t* buf, size_t len);

  /**
   * Seek to a position in a file handle.
   *
   * Returns the resulting position, or -1 on error.
   */
  int64_t (*seek)(void* handle, int64_t offset, int32_t whence);
} AuroraOverlayCallbacks;

/**
 * \brief Specify callbacks for overlaid files.
 */
void aurora_dvd_overlay_callbacks(const AuroraOverlayCallbacks* callbacks);

/**
 * \brief Specify a set of overlay files to be used by the DVD layer.
 *
 * Calling this function immediately applies the new files and rebuilds the FST, replacing any
 * previously specified set. It may be called again at runtime; FST reads and overlay opens are
 * serialized against the rebuild, and previously assigned EntryNums are stable per path. Files
 * removed by a re-registration fail to open from then on (or revert to the underlying DVD file);
 * already-open handles are unaffected. Only call from one thread at a time.
 *
 * This function must be called *after* aurora_dvd_overlay_callbacks.
 *
 * @param files Array of AuroraOverlayFiles, one for every file being overlaid.
 * @param nFiles Amount of files in the array.
 * @param outEntryNums Optional output array receiving one EntryNum per input file. Unaccepted files receive -1.
 */
void aurora_dvd_overlay_files(const AuroraOverlayFile* files, size_t nFiles, s32* outEntryNums);

/**
 * \brief Gets the amount of FST entries present on the loaded game disc.
 *
 * This does not take overlay files into account.
 */
s32 aurora_dvd_base_entry_count();

/**
 * \brief Gets the EntryNum of the file at an absolute path on the loaded game disc, ignoring overlays.
 *
 * Returns -1 if the disc has no file at that path.
 */
s32 aurora_dvd_base_entry_num(const char* path);

/**
 * \brief Gets the size in bytes of a file on the loaded game disc, ignoring overlays.
 *
 * Returns -1 if baseEntryNum is not a file on the disc.
 */
int64_t aurora_dvd_base_file_size(s32 baseEntryNum);

/**
 * \brief Reads from a file on the loaded game disc, ignoring overlays.
 *
 * May be called from any thread. Returns the number of bytes read (less than len at the end of the file), or -1 on
 * error.
 */
int64_t aurora_dvd_read_base_file(s32 baseEntryNum, void* buf, size_t len, uint64_t offset);

#ifdef __cplusplus
}
#endif

#endif
