#pragma once

// Generated from the raw ALine order 1 DTO with singleton Q orbits.
static_assert(PET_N == 2, "Selected factor table does not match SEMGPU2_SEISMIC_ORDER");

static __constant__ const FactorTables factor = {
    {
        0x1.0000000000000p+0, 0x1.0000000000000p+0,
    },
    {
        0x1.0000000000000p+0, 0x0.0p+0,
        0x0.0p+0, 0x1.0000000000000p+0,
    },
    {
        -0x1.0000000000000p-1, 0x1.0000000000000p-1,
        -0x1.0000000000000p-1, 0x1.0000000000000p-1,
    },
};
