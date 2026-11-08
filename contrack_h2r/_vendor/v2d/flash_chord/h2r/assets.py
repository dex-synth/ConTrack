# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def numbers(values):
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).reshape(-1))


def portable_xml(source, destination):
    """Resolve mesh paths before moving an XML; leave the source package untouched."""
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    if compiler is not None:
        compiler.set("meshdir", "")
    for mesh in root.findall("asset/mesh"):
        resolved = (Path(source).parent / meshdir / mesh.attrib["file"]).resolve()
        if not resolved.is_relative_to(Path(source).parent.resolve()):
            raise ValueError(f"Mesh escapes its common package: {resolved}")
        # Assets and copied source packages live below the same preparation root.
        mesh.set("file", os.path.relpath(resolved, Path(destination).parent))
    return root
