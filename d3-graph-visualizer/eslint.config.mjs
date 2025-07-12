import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    rules: {
      // Suppress unreachable code warnings from third-party libraries
      "no-unreachable": "off",
      // Suppress other common warnings from node_modules
      "no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
    },
  },
  {
    // Ignore warnings from node_modules
    ignores: ["node_modules/**/*"],
  },
];

export default eslintConfig;
