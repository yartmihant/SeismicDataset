"""Материалы ECVB в FC v3, изотропный закон Гука в параметрах Vp, Vs и ρ."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

import numpy as np
from fc_model import FCMaterial, FCModel

DENSITY_KG_M3 = 1000.0


def _constants(material: FCMaterial) -> dict[str, float]:
    values: dict[str, float] = {}
    for groups, law in ((material.properties["common"], "USUAL"), (material.properties["elasticity"], "HOOK")):
        for properties in groups:
            for prop in properties:
                data = prop.data.value.data
                if (prop.type != law or prop.data.type != "CONSTANT" or
                        not isinstance(data, np.ndarray) or data.shape != (1,)):
                    raise ValueError(f"Invalid constant in FC material {material.id}: {prop.name}")
                values[str(prop.name)] = float(data[0])
    return values


def read_materials(model: FCModel, cad: dict) -> dict[int, tuple[float, float]]:
    """Пройти по реальным связям блок → материал и сверить Vp, Vs, ρ с исходной картой."""
    velocities = {r["id"]: (r["vp"], r["vs"]) for r in cad["regions"]}
    if (len(velocities) != len(cad["regions"]) or set(model.materials) != set(velocities)
            or set(model.blocks) != set(velocities)):
        raise ValueError("FC regions, blocks and materials differ")
    result: dict[int, tuple[float, float]] = {}
    for region, (vp, vs) in velocities.items():
        assignment = model.blocks[region].material_id
        if assignment != region:
            raise ValueError(f"Incorrect material assignment in FC block {region}")
        properties = _constants(model.materials[assignment])
        if properties != {"DENSITY": DENSITY_KG_M3, "VP": vp, "VS": vs}:
            raise ValueError(f"Incorrect density or velocities in FC block {region}")
        result[region] = (properties["VP"], properties["VS"])
    return result


def write_material_fc(source: Path, cad: dict, target: Path) -> dict:
    """Сохранить исходные массивы QUAD8 и присвоить каждому блоку материал его области CAD."""
    document = json.loads(source.read_text(encoding="utf-8"))
    velocities = {r["id"]: (r["vp"], r["vs"]) for r in cad["regions"]}
    blocks = document["blocks"]
    if (not velocities or len(velocities) != len(cad["regions"]) or
            len(blocks) != len(velocities) or {b["id"] for b in blocks} != set(velocities)):
        raise ValueError("FC blocks and CAD regions must be in one-to-one correspondence")
    materials = []
    for block in blocks:
        region = block["id"]
        vp, vs = velocities[region]
        if not (isfinite(vp) and isfinite(vs) and vp > 0 and vs > 0 and vp**2 > 4 * vs**2 / 3):
            raise ValueError(f"Nonphysical isotropic velocities in region {region}: {(vp, vs)}")
        material = FCMaterial(id=region, name=f"ECVB{cad['model']:05d}_region_{region}")
        material.add_property("common", "DENSITY", DENSITY_KG_M3, "USUAL")
        material.add_property("elasticity", "VP", vp, "HOOK")
        material.add_property("elasticity", "VS", vs, "HOOK")
        materials.append(material.encode())
        block["material_id"] = region
    document["materials"] = materials
    target.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                      encoding="utf-8")

    saved = json.loads(target.read_text(encoding="utf-8"))
    if saved["mesh"] != document["mesh"]:
        raise ValueError("Mesh payload changed while assigning FC materials")
    read_materials(FCModel.decode(saved), cad)
    return {"law": "HOOK", "parameters": ["VP", "VS", "DENSITY"], "density_kg_m3": DENSITY_KG_M3,
            "block_material_ids_equal_region_ids": True, "materials": len(materials),
            "mesh_payload_identical": True, "fc_velocities_exact": True}
