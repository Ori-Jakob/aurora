#include <gtest/gtest.h>

#include "render/json.hpp"

#include <cstdint>
#include <string>

namespace aurora::render::json {
namespace {
TEST(RenderJsonTest, WriteReadRoundTrip) {
  const MutDoc doc{yyjson_mut_doc_new(nullptr)};
  ASSERT_TRUE(doc);
  yyjson_mut_val* root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_int(doc.get(), root, "api", 1);
  yyjson_mut_obj_add_str(doc.get(), root, "name", "aurora");
  const int64_t values[] = {1, 2, 3};
  yyjson_mut_obj_add_val(doc.get(), root, "list", yyjson_mut_arr_with_sint64(doc.get(), values, 3));

  const std::string text = write(doc.get(), false);
  EXPECT_EQ(text, R"({"api":1,"name":"aurora","list":[1,2,3]})");

  std::string error;
  const Doc parsed = read(text, &error);
  ASSERT_TRUE(parsed) << error;
  yyjson_val* parsedRoot = yyjson_doc_get_root(parsed.get());
  EXPECT_EQ(yyjson_get_int(yyjson_obj_get(parsedRoot, "api")), 1);
  EXPECT_EQ(yyjson_arr_size(yyjson_obj_get(parsedRoot, "list")), 3u);

  const std::string pretty = write(doc.get(), true);
  ASSERT_FALSE(pretty.empty());
  EXPECT_EQ(pretty.back(), '\n');
}

TEST(RenderJsonTest, ReadReportsErrorPosition) {
  std::string error;
  EXPECT_FALSE(read("{\"a\":}", &error));
  EXPECT_NE(error.find("at byte 5"), std::string::npos) << error;
}
} // namespace
} // namespace aurora::render::json
