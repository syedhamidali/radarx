import importlib
import pkgutil


def scan_radarx_modules():
    import radarx

    base_pkg = radarx
    print(f"📦 Scanning '{base_pkg.__name__}' package...\n")

    implemented = []

    for loader, module_name, is_pkg in pkgutil.iter_modules(base_pkg.__path__):
        full_module = f"{base_pkg.__name__}.{module_name}"
        try:
            mod = importlib.import_module(full_module)
            exported = getattr(mod, "__all__", [])
            implemented.append((module_name, exported))
        except Exception as e:
            print(f"⚠️  Failed to import {full_module}: {e}")
            implemented.append((module_name, []))

    # Print summary
    print("🔍 Module Overview:")
    for name, items in implemented:
        status = "✅" if items else "⏳"
        print(
            f"{status} {name:12} | {len(items)} items exported: {items if items else 'None'}"
        )

    return implemented


if __name__ == "__main__":
    scan_radarx_modules()
