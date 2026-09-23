"""
One-time setup: downloads and installs open-source Argos Translate language
packages for English <-> Spanish/French/Hindi.

Run this once, in an environment with normal internet access:

    python argos_setup.py

After this, translation.py will automatically pick up the installed
packages and use ArgosTranslator (real neural MT) instead of the offline
glossary fallback -- no code changes needed.

Note: this sandbox's network egress is restricted to a small allow-list of
package-registry domains, which does not include Argos Translate's model
host (argos-net.com), so this script won't succeed inside this sandbox.
It's provided for you to run wherever this project is actually deployed.
"""
import argostranslate.package as package

PAIRS = [("en", "es"), ("es", "en"), ("en", "fr"), ("fr", "en"), ("en", "hi"), ("hi", "en")]


def main():
    print("Updating Argos Translate package index...")
    package.update_package_index()
    available = package.get_available_packages()

    for src, tgt in PAIRS:
        match = next((p for p in available if p.from_code == src and p.to_code == tgt), None)
        if not match:
            print(f"  ! No package found for {src} -> {tgt}, skipping")
            continue
        already = any(
            l.code == src for l in package.get_installed_languages()
        ) and any(l.code == tgt for l in package.get_installed_languages())
        print(f"  Installing {src} -> {tgt} ...")
        path = match.download()
        package.install_from_path(path)

    print("Done. Installed languages:", [l.code for l in package.get_installed_languages()])


if __name__ == "__main__":
    main()
