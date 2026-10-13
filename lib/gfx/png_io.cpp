#include "png_io.hpp"

#include "../io.hpp"
#include "png.h"

#include <cstring>

static aurora::Module Log("aurora::gfx::png");

namespace aurora::gfx::png {

struct PngStructs {
  png_structp pStruct = nullptr;
  png_infop pInfo = nullptr;

  ~PngStructs() {
    png_destroy_read_struct(&pStruct, &pInfo, nullptr);
  }
};

struct MemoryCursor {
  ArrayRef<uint8_t> bytes;
  size_t pos = 0;
};

static void readPngData(png_structp png, png_bytep data, const size_t length) {
  auto* cursor = static_cast<MemoryCursor*>(png_get_io_ptr(png));
  if (length > cursor->bytes.size() - cursor->pos) {
    png_error(png, "unexpected end of data");
  }
  std::memcpy(data, cursor->bytes.data() + cursor->pos, length);
  cursor->pos += length;
}

std::optional<ConvertedTexture> parse_png_bytes(ArrayRef<uint8_t> bytes) noexcept {
  PngStructs structs{};
  MemoryCursor cursor{bytes};

  structs.pStruct = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!structs.pStruct) {
    Log.error("png_create_read_struct failed");
    return std::nullopt;
  }

  structs.pInfo = png_create_info_struct(structs.pStruct);
  if (!structs.pInfo) {
    Log.error("png_create_info_struct failed");
    return std::nullopt;
  }

  // I'm scared of putting any locals after that setjmp.
  std::vector<png_bytep> rowPointers;
  ByteBuffer imageData{};
  png_uint_32 width, height;
  int bit_depth, color_type, interlace_type, compression_type, filter_type;
  size_t rowBytes;
  int i;

  if (setjmp(png_jmpbuf(structs.pStruct))) {
    Log.error("libpng encountered an error");
    return std::nullopt;
  }

  png_set_read_fn(structs.pStruct, &cursor, readPngData);
  png_read_info(structs.pStruct, structs.pInfo);

  if (!png_get_IHDR(structs.pStruct, structs.pInfo, &width, &height, &bit_depth, &color_type, &interlace_type, &compression_type, &filter_type)) {
    Log.error("libpng unable to read IHDR");
    return std::nullopt;
  }

  // Always read as RGBA8.
  png_set_gray_to_rgb(structs.pStruct);
  png_set_filler(structs.pStruct, 0xFF, PNG_FILLER_AFTER);
  png_set_expand(structs.pStruct);
  png_set_strip_16(structs.pStruct);

  png_read_update_info(structs.pStruct, structs.pInfo);
  rowBytes = png_get_rowbytes(structs.pStruct, structs.pInfo);
  rowPointers.resize(height);

  imageData.append_zeroes(rowBytes * height);

  for (i = 0; i < height; i++) {
    rowPointers[i] = imageData.data() + i * rowBytes;
  }

  png_read_image(structs.pStruct, rowPointers.data());
  png_read_end(structs.pStruct, nullptr);

  return ConvertedTexture{
    .format = wgpu::TextureFormat::RGBA8Unorm,
    .width = width,
    .height = height,
    .mips = 1,
    .data = std::move(imageData)
  };
}

std::optional<ConvertedTexture>
load_png_file(const std::filesystem::path& path) noexcept {
  const auto bytes = io::read_file(path);
  if (!bytes.has_value()) {
    Log.error("failed to open file: {}", io::fs_path_to_string(path));
    return std::nullopt;
  }
  return parse_png_bytes(*bytes);
}

static void writePngData(png_structp png, png_bytep data, const size_t length) {
  static_cast<ByteBuffer*>(png_get_io_ptr(png))->append(data, length);
}

static void flushPngData(png_structp) {}

struct PngWriteStructs {
  png_structp pStruct = nullptr;
  png_infop pInfo = nullptr;

  ~PngWriteStructs() { png_destroy_write_struct(&pStruct, &pInfo); }
};

ByteBuffer encode_rgba8_png(uint32_t width, uint32_t height, ArrayRef<uint8_t> pixels) noexcept {
  if (width == 0 || height == 0 || pixels.size() != static_cast<size_t>(width) * height * 4) {
    Log.error("encode_rgba8_png: {} bytes do not form a {}x{} RGBA8 image", pixels.size(), width, height);
    return {};
  }
  PngWriteStructs structs{};
  structs.pStruct = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!structs.pStruct) {
    Log.error("png_create_write_struct failed");
    return {};
  }
  structs.pInfo = png_create_info_struct(structs.pStruct);
  if (!structs.pInfo) {
    Log.error("png_create_info_struct failed");
    return {};
  }

  ByteBuffer encoded;
  std::vector<png_const_bytep> rowPointers(height);
  for (uint32_t y = 0; y < height; ++y) {
    rowPointers[y] = pixels.data() + static_cast<size_t>(y) * width * 4;
  }
  if (setjmp(png_jmpbuf(structs.pStruct))) {
    Log.error("libpng encountered an error");
    return {};
  }

  png_set_write_fn(structs.pStruct, &encoded, writePngData, flushPngData);
  png_set_IHDR(structs.pStruct, structs.pInfo, width, height, 8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_set_compression_level(structs.pStruct, 6);
  png_write_info(structs.pStruct, structs.pInfo);
  png_write_image(structs.pStruct, const_cast<png_bytepp>(rowPointers.data()));
  png_write_end(structs.pStruct, nullptr);
  return encoded;
}

bool write_rgba8_png(const std::filesystem::path& path, uint32_t width, uint32_t height,
                     ArrayRef<uint8_t> pixels) noexcept {
  const auto encoded = encode_rgba8_png(width, height, pixels);
  if (encoded.empty()) {
    return false;
  }
  if (!io::write_file_atomic(path, {encoded.data(), encoded.size()})) {
    Log.error("failed to write {}", io::fs_path_to_string(path));
    return false;
  }
  return true;
}
}
