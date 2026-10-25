#include "manifest.hpp"

namespace aurora::render::material {
namespace {
// Frozen copies of the C1 schemas; a test pins them to tests/fixtures/materials
constexpr std::string_view BaseSchemaText = R"schema({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://aurora.invalid/schemas/texture-materials/1/materials.schema.json",
  "title": "aurora texture material manifest, version 1",
  "type": "object",
  "required": ["version"],
  "additionalProperties": false,
  "properties": {
    "$schema": { "type": "string" },
    "version": { "x-c1-rule": "MANIFEST-VERSION", "const": 1 },
    "grassTypes": {
      "type": "object",
      "propertyNames": { "$ref": "#/$defs/grassTypeName" },
      "additionalProperties": { "$ref": "#/$defs/grassType" }
    },
    "materials": {
      "type": "object",
      "propertyNames": { "$ref": "#/$defs/materialKey" },
      "additionalProperties": { "$ref": "#/$defs/material" }
    }
  },
  "$defs": {
    "unit": { "type": "number", "minimum": 0, "maximum": 1 },
    "color": { "type": "string", "pattern": "^#[0-9A-Fa-f]{6}$" },
    "meterRange": {
      "type": "array", "minItems": 2, "maxItems": 2,
      "items": { "type": "number", "exclusiveMinimum": 0, "maximum": 4 }
    },
    "grassTypeName": { "type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$" },
    "materialKey": {
      "type": "string",
      "allOf": [
        { "x-c1-rule": "KEY-TEXTURE-WILDCARD", "not": { "pattern": "^tex1_[0-9]+x[0-9]+(_m)?_\\$(_|$)" } },
        { "x-c1-rule": "KEY-TLUT-FORMAT",
          "not": { "pattern": "^tex1_[1-9][0-9]{0,3}x[1-9][0-9]{0,3}_[0-9a-f]{16}(_(8|9|10)|_([0-9a-f]{16}|\\$)_([0-6]|14|65|67|70|78))$" } },
        { "x-c1-rule": "KEY-SYNTAX",
          "pattern": "^tex1_[1-9][0-9]{0,3}x[1-9][0-9]{0,3}_[0-9a-f]{16}(_([0-9a-f]{16}|\\$)_(8|9|10)|_([0-6]|14|65|67|70|78))$" }
      ]
    },
    "mapPath": {
      "x-c1-rule": "MAP-PATH", "type": "string", "maxLength": 240,
      "pattern": "^([A-Za-z0-9_$-][A-Za-z0-9_.$-]*/)*[A-Za-z0-9_$-][A-Za-z0-9_.$-]*\\.[Dd][Dd][Ss]$"
    },
    "material": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "flags": { "type": "array", "uniqueItems": true, "items": { "enum": ["stochastic", "noPbr", "cutout", "unlit"] } },
        "class": { "type": "string", "maxLength": 64, "pattern": "^[a-z][a-z0-9_]*(/[a-z][a-z0-9_]*)*$" },
        "vertexColor": { "enum": ["tint", "bakedLight", "ignore"] },
        "roughness": { "$ref": "#/$defs/unit" },
        "metalness": { "$ref": "#/$defs/unit" },
        "heightScale": { "type": "number", "minimum": 0, "maximum": 1 },
        "normalStrength": { "type": "number", "minimum": 0, "maximum": 4 },
        "normalFlipY": { "type": "boolean", "default": false },
        "stochastic": {
          "type": "object", "additionalProperties": false,
          "properties": {
            "contrast": { "$ref": "#/$defs/unit" },
            "rotation": { "$ref": "#/$defs/unit" },
            "scale": { "type": "number", "exclusiveMinimum": 0, "maximum": 16 },
            "exponent": { "type": "number", "minimum": 1, "maximum": 32 }
          }
        },
        "maps": {
          "type": "object", "additionalProperties": false,
          "properties": {
            "albedo": { "$ref": "#/$defs/mapPath" }, "delight": { "$ref": "#/$defs/mapPath" },
            "nrm": { "$ref": "#/$defs/mapPath" }, "orm": { "$ref": "#/$defs/mapPath" },
            "emit": { "$ref": "#/$defs/mapPath" }, "grassmask": { "$ref": "#/$defs/mapPath" },
            "mat": { "$ref": "#/$defs/mapPath" }
          }
        },
        "grass": {
          "type": "object", "additionalProperties": false,
          "properties": {
            "type": { "$ref": "#/$defs/grassTypeName" },
            "density": { "$ref": "#/$defs/unit" },
            "densitySource": { "enum": ["constant", "vtxAlpha", "vtxAlphaInv", "tev"] },
            "heightScale": { "type": "number", "exclusiveMinimum": 0, "maximum": 4 },
            "maxSlopeDeg": { "type": "number", "minimum": 0, "maximum": 90 },
            "aquatic": { "type": "boolean", "default": false },
            "allowInterior": { "type": "boolean", "default": false },
            "useGroundAlbedo": { "type": "boolean", "default": true }
          }
        },
        "displacement": {
          "type": "object", "additionalProperties": false,
          "properties": {
            "amplitude": { "type": "number", "minimum": 0, "maximum": 64 },
            "center": { "type": "number", "minimum": 0, "maximum": 1, "default": 1 },
            "tess": { "type": "boolean", "default": true },
            "heightBlend": { "enum": ["orm", "vertexAlpha", "first", "max"], "default": "orm" }
          }
        },
        "fakeShadow": {
          "type": "object", "additionalProperties": false,
          "properties": {
            "neutral": { "enum": ["white", "clear", "skip"] },
            "gate": { "enum": ["light", "cloud", "ao"] },
            "scope": {
              "type": "array", "minItems": 1, "uniqueItems": true,
              "items": { "enum": ["STATIC", "DYNAMIC", "SKINNED", "TERRAIN", "FOLIAGE", "WATER", "SKY", "PARTICLE",
                                  "DECAL", "UI", "PLAYER", "EMISSIVE", "OVERLAY"] }
            }
          }
        },
        "_source": { "type": "string", "minLength": 1, "maxLength": 1024 },
        "_provenance": {
          "type": "object", "additionalProperties": false,
          "properties": {
            "albedo": { "$ref": "#/$defs/provenanceRecord" }, "delight": { "$ref": "#/$defs/provenanceRecord" },
            "nrm": { "$ref": "#/$defs/provenanceRecord" }, "orm": { "$ref": "#/$defs/provenanceRecord" },
            "emit": { "$ref": "#/$defs/provenanceRecord" }, "grassmask": { "$ref": "#/$defs/provenanceRecord" },
            "mat": { "$ref": "#/$defs/provenanceRecord" }
          }
        }
      }
    },
    "provenanceRecord": {
      "type": "object",
      "properties": {
        "tool": { "type": "string" }, "version": { "type": "string" },
        "inputs": { "type": "array", "items": { "type": "string" } },
        "license": { "type": "string" },
        "sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" }
      }
    },
    "grassType": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "height": { "$ref": "#/$defs/meterRange" },
        "width": { "$ref": "#/$defs/meterRange" },
        "densityPerM2": { "type": "number", "exclusiveMinimum": 0, "maximum": 10000 },
        "tilt": { "$ref": "#/$defs/unit" },
        "bend": { "$ref": "#/$defs/unit" },
        "clump": {
          "type": "object", "additionalProperties": false,
          "properties": {
            "size": { "type": "number", "exclusiveMinimum": 0, "maximum": 16 },
            "heightVariance": { "$ref": "#/$defs/unit" },
            "colorVariance": { "$ref": "#/$defs/unit" },
            "lean": { "$ref": "#/$defs/unit" }
          }
        },
        "rootColor": { "$ref": "#/$defs/color" },
        "tipColor": { "$ref": "#/$defs/color" },
        "dryTipColor": { "$ref": "#/$defs/color" },
        "colorFrom": { "$ref": "#/$defs/materialKey" },
        "translucency": { "$ref": "#/$defs/unit" },
        "roughness": { "$ref": "#/$defs/unit" },
        "specular": { "$ref": "#/$defs/unit" },
        "lodVerts": {
          "type": "array", "minItems": 1, "maxItems": 4, "default": [15, 7, 3],
          "items": { "enum": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] }
        },
        "flags": { "type": "array", "uniqueItems": true, "items": { "enum": ["CAST_SHELL", "CAST_FULL", "AQUATIC"] } }
      }
    }
  }
}
)schema";

constexpr std::string_view Layer1SchemaText = R"schema({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://aurora.invalid/schemas/texture-materials/1/materials.layer1.schema.json",
  "title": "aurora texture material manifest, version 1, shipped game layer",
  "allOf": [
    { "$ref": "materials.schema.json" },
    { "properties": { "materials": { "additionalProperties": { "$ref": "#/$defs/layer1Material" } } } }
  ],
  "$defs": {
    "layer1Material": {
      "propertyNames": {
        "x-c1-rule": "LAYER1-FORBIDDEN-MEMBER",
        "enum": ["flags", "class", "vertexColor", "stochastic", "grass", "displacement", "fakeShadow", "_source"]
      },
      "properties": {
        "fakeShadow": { "x-c1-rule": "LAYER1-FAKESHADOW-INCOMPLETE", "required": ["neutral", "gate", "scope"] },
        "_source": {
          "x-c1-rule": "LAYER1-SOURCE", "maxLength": 256,
          "pattern": "^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*(:[A-Za-z0-9_.-]+)?( (mat|blend|texmap|room|stage):[A-Za-z0-9_.,-]+)*$"
        }
      }
    }
  }
}
)schema";
} // namespace

std::string_view embedded_schema(std::string_view name) noexcept {
  if (name == "materials.schema.json") {
    return BaseSchemaText;
  }
  if (name == "materials.layer1.schema.json") {
    return Layer1SchemaText;
  }
  return {};
}
} // namespace aurora::render::material
