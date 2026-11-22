#pragma once

// Generated from the raw ALine order 3 DTO with singleton Q orbits.
static_assert(PET_N == 4, "Selected factor table does not match SEMGPU2_SEISMIC_ORDER");

static __constant__ const FactorTables factor = {
    {
        0x1.5555555555554p-3, 0x1.aaaaaaaaaaaaap-1, 0x1.aaaaaaaaaaaabp-1, 0x1.5555555555554p-3,
    },
    {
        0x1.0000000000000p+0, -0x1.0000000000000p-53, 0x0.0p+0, 0x0.0p+0,
        -0x1.8000000000000p-55, 0x1.0000000000000p+0, -0x1.7f2409432ec20p-54, -0x1.0000000000000p-55,
        -0x1.0000000000000p-55, -0x1.3f9204a197610p-53, 0x1.0000000000001p+0, -0x1.8000000000000p-55,
        0x0.0p+0, -0x1.0000000000000p-53, -0x1.0000000000000p-53, 0x1.0000000000000p+0,
    },
    {
        -0x1.8000000000000p+1, 0x1.02e2ac13ef8eap+2, -0x1.8b8ab04fbe3a6p+0, 0x1.0000000000000p-1,
        -0x1.9e3779b97f4a8p-1, -0x1.145aab3b2c9f7p-53, 0x1.1e3779b97f4a9p+0, -0x1.3c6ef372fe950p-2,
        0x1.3c6ef372fe950p-2, -0x1.1e3779b97f4a9p+0, 0x1.145aab3b2c9f7p-53, 0x1.9e3779b97f4a8p-1,
        -0x1.0000000000000p-1, 0x1.8b8ab04fbe3a6p+0, -0x1.02e2ac13ef8eap+2, 0x1.8000000000000p+1,
    },
};
