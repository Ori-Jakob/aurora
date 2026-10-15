#pragma once

#include <yyjson.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>

namespace aurora::render::json {
struct DocDeleter {
  void operator()(yyjson_doc* doc) const noexcept { yyjson_doc_free(doc); }
};
struct MutDocDeleter {
  void operator()(yyjson_mut_doc* doc) const noexcept { yyjson_mut_doc_free(doc); }
};
using Doc = std::unique_ptr<yyjson_doc, DocDeleter>;
using MutDoc = std::unique_ptr<yyjson_mut_doc, MutDocDeleter>;

inline Doc read(std::string_view text, std::string* error = nullptr) {
  yyjson_read_err err{};
  // yyjson only writes through the pointer with YYJSON_READ_INSITU
  auto* doc = yyjson_read_opts(const_cast<char*>(text.data()), text.size(), YYJSON_READ_NOFLAG, nullptr, &err);
  if (doc == nullptr && error != nullptr) {
    *error = std::string{err.msg != nullptr ? err.msg : "unknown error"} + " at byte " + std::to_string(err.pos);
  }
  return Doc{doc};
}

inline std::string write(const yyjson_mut_doc* doc, bool pretty) {
  const yyjson_write_flag flags = pretty ? YYJSON_WRITE_PRETTY_TWO_SPACES | YYJSON_WRITE_NEWLINE_AT_END : 0;
  size_t length = 0;
  char* text = yyjson_mut_write_opts(doc, flags, nullptr, &length, nullptr);
  if (text == nullptr) {
    return {};
  }
  std::string out{text, length};
  std::free(text);
  return out;
}
} // namespace aurora::render::json
