import numpy as np

from src.environment.data.zone_config import LOCAL_AREAS

TRAVEL_MATRIX = np.full(
    (len(LOCAL_AREAS), len(LOCAL_AREAS)),
    3,
    dtype=np.float32
)

np.fill_diagonal(TRAVEL_MATRIX, 0)

REGIONS = {
    "Downtown Core": [
        "Downtown",
        "West End",
        "Strathcona"
    ],

    "East Vancouver": [
        "Grandview-Woodland",
        "Hastings-Sunrise",
        "Kensington-Cedar Cottage",
        "Renfrew-Collingwood",
        "Killarney",
        "Victoria-Fraserview",
        "Sunset"
    ],

    "Central Vancouver": [
        "Mount Pleasant",
        "Fairview",
        "Riley Park",
        "South Cambie",
        "Oakridge"
    ],

    "West Side": [
        "Arbutus Ridge",
        "Dunbar-Southlands",
        "Kerrisdale",
        "Kitsilano",
        "Shaughnessy",
        "West Point Grey",
        "Marpole"
    ]
}

for region in REGIONS.values():

    ids = [
        LOCAL_AREAS.index(area)
        for area in region
    ]

    for i in ids:
        for j in ids:
            if i != j:
                TRAVEL_MATRIX[i, j] = 1