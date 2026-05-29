"""Pre-deploy preflight: catch undefined-name (NameError) regressions
without running the code.

py_compile only catches syntax errors. The 2026-05-29 crash-loop was a
NameError inside a function body (a renamed local still referenced under
its old name) — py_compile passed it, and it took down the live daemon
for ~23h. pyflakes would have caught it but isn't always installed.

This is a focused, stdlib-only (ast) undefined-name checker. It walks
each function scope and flags any Name used in Load context that is not
bound in:
  - that function's locals (args, assignments, for/with/except targets,
    comprehension targets, walrus, global/nonlocal declarations)
  - any enclosing function scope
  - module globals (imports, module-level assignments, def/class names)
  - Python builtins

Conservative by design: when in doubt it does NOT flag, to avoid
false-positive deploy blocks. It will miss some real errors that
pyflakes catches, but it reliably catches the "renamed variable, stale
reference" class that bit us.

Usage:
  python deploy/preflight.py slim_daemon.py weather_bot/*.py
  # exit 0 = clean, exit 1 = undefined names found
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

_BUILTINS = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__builtins__",
    "__spec__", "__loader__", "__annotations__",
}


def _collect_bound_names(body_nodes) -> set[str]:
    """Names bound directly in a scope, given an ITERABLE of body
    statements (not the function/module node itself). Non-recursive into
    nested FunctionDef/ClassDef bodies, but DOES descend into control
    flow (if/for/while/try/with)."""
    bound: set[str] = set()

    class Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, n):
            bound.add(n.name)  # the def name is bound in the enclosing scope
            # do NOT recurse into the nested function body

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, n):
            bound.add(n.name)

        def visit_Name(self, n):
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                bound.add(n.id)

        def visit_arg(self, n):
            bound.add(n.arg)

        def visit_Global(self, n):
            bound.update(n.names)

        def visit_Nonlocal(self, n):
            bound.update(n.names)

        def visit_Import(self, n):
            for a in n.names:
                bound.add(a.asname or a.name.split(".")[0])

        def visit_ImportFrom(self, n):
            for a in n.names:
                bound.add(a.asname or a.name)

        def visit_ExceptHandler(self, n):
            if n.name:
                bound.add(n.name)
            self.generic_visit(n)

    c = Collector()
    for n in body_nodes:
        c.visit(n)
    return bound


def _function_scopes(tree: ast.AST):
    """Yield (FunctionDef node, set-of-enclosing-bound-names) for every
    function, where enclosing includes module globals + outer functions."""
    module_bound = _collect_bound_names(tree.body)

    def walk(node, enclosing: set[str]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local = _collect_bound_names(child.body)
                # args of this function
                for a in (child.args.posonlyargs + child.args.args
                          + child.args.kwonlyargs):
                    local.add(a.arg)
                if child.args.vararg:
                    local.add(child.args.vararg.arg)
                if child.args.kwarg:
                    local.add(child.args.kwarg.arg)
                scope = enclosing | local
                yield child, scope
                yield from walk(child, scope)
            elif isinstance(child, ast.ClassDef):
                cls_bound = enclosing | _collect_bound_names(child.body)
                yield from walk(child, cls_bound)
            else:
                yield from walk(child, enclosing)

    yield from walk(tree, module_bound | _BUILTINS)


def _loaded_names(func: ast.AST):
    """Yield (name, lineno) for every Name used in Load context within
    this function, NOT descending into nested function bodies (those are
    checked with their own scope)."""
    skip: set[int] = set()
    for n in ast.walk(func):
        if n is not func and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(n):
                skip.add(id(sub))
    for n in ast.walk(func):
        if id(n) in skip:
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            yield n.id, n.lineno


def check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as e:
        return [f"{path}:{e.lineno}: SyntaxError: {e.msg}"]

    problems: list[str] = []
    for func, scope in _function_scopes(tree):
        for name, lineno in _loaded_names(func):
            if name not in scope:
                problems.append(
                    f"{path}:{lineno}: undefined name '{name}' "
                    f"in function '{func.name}'"
                )
    return problems


def main(argv: list[str]) -> int:
    files: list[Path] = []
    for arg in argv:
        p = Path(arg)
        if p.is_dir():
            files.extend(p.rglob("*.py"))
        else:
            files.append(p)
    if not files:
        print("usage: python deploy/preflight.py <file-or-dir> ...")
        return 2

    all_problems: list[str] = []
    for f in sorted(set(files)):
        all_problems.extend(check_file(f))

    if all_problems:
        print("PREFLIGHT FAILED — undefined names found:")
        for p in all_problems:
            print(f"  {p}")
        return 1
    print(f"PREFLIGHT OK — {len(files)} files checked, no undefined names.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
