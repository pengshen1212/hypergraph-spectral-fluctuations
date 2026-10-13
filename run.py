"""Run preserved experiment sources with repository-relative path literals."""

import argparse
import ast
import importlib.abc
import importlib.machinery
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "paper/paper1/experiments/prob2-relation-error"
PATH_PREFIX = "C:/Users/Peng1/OneDrive/\u6587\u6863/ChatGPT/hypergraph"


class RelativePaths(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            normalized = node.value.replace("\\", "/")
            if normalized == PATH_PREFIX or normalized.startswith(PATH_PREFIX + "/"):
                relative = normalized[len(PATH_PREFIX):].lstrip("/")
                return ast.copy_location(ast.Constant(str(ROOT / relative)), node)
        return node


def compile_source(data, filename, optimize=-1):
    tree = RelativePaths().visit(ast.parse(data, filename=str(filename)))
    return compile(ast.fix_missing_locations(tree), str(filename), "exec", optimize=optimize)


class RelativeLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        return compile_source(Path(self.path).read_bytes(), self.path)


class RelativeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if not spec or not spec.origin or not spec.origin.endswith(".py"):
            return None
        origin = Path(spec.origin).resolve()
        if origin.is_relative_to(ROOT) and origin != Path(__file__).resolve():
            spec.loader = RelativeLoader(fullname, str(origin))
            return spec
        return None


def install():
    sys.meta_path.insert(0, RelativeFinder())
    for directory in (BASE, BASE / "upgrade", BASE / "native_risk"):
        sys.path.insert(0, str(directory))


def test():
    modules = ["test_node_influence", "test_conditional_moments", "test_exact_certificate",
               "test_finite_sample", "test_verified_norm", "test_spectral_supplement",
               "test_n4_runner", "test_n4_repair", "test_n4r_execution"]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name)
                               for name in modules)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "test", "script"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    install()
    if args.command == "verify":
        from verify_artifacts import verify
        verify()
        return 0
    if args.command == "test":
        return test()
    if not args.arguments:
        parser.error("script requires a repository-relative .py path")
    target = (ROOT / args.arguments[0]).resolve()
    if not target.is_relative_to(ROOT) or target.suffix != ".py" or not target.is_file():
        parser.error("script path must identify a .py file within the repository")
    sys.argv = [str(target), *args.arguments[1:]]
    sys.path.insert(0, str(target.parent))
    os.chdir(target.parent)
    scope = {"__name__": "__main__", "__file__": str(target), "__package__": None}
    exec(compile_source(target.read_bytes(), target), scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
