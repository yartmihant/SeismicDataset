#pragma once

// Generated from the raw ALine order 2 DTO with singleton Q orbits.
static_assert(PET_N == 3, "Selected factor table does not match SEMGPU2_SEISMIC_ORDER");

static __constant__ const FactorTables factor = {
    {
        0x1.5555555555558p-2, 0x1.5555555555555p+0, 0x1.5555555555556p-2,
    },
    {
        0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0,
        0x1.8000000000000p-54, 0x1.0000000000000p+0, 0x1.0000000000000p-55,
        0x1.0000000000000p-54, 0x0.0p+0, 0x1.0000000000000p+0,
    },
    {
        -0x1.8000000000000p+0, 0x1.0000000000000p+1, -0x1.0000000000000p-1,
        -0x1.0000000000000p-1, 0x0.0p+0, 0x1.0000000000000p-1,
        0x1.0000000000000p-1, -0x1.0000000000000p+1, 0x1.8000000000000p+0,
    },
};
