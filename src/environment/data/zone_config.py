LOCAL_AREAS = [
    "Arbutus Ridge",
    "Downtown",
    "Dunbar-Southlands",
    "Fairview",
    "Grandview-Woodland",
    "Hastings-Sunrise",
    "Kensington-Cedar Cottage",
    "Kerrisdale",
    "Killarney",
    "Kitsilano",
    "Marpole",
    "Mount Pleasant",
    "Oakridge",
    "Renfrew-Collingwood",
    "Riley Park",
    "Shaughnessy",
    "South Cambie",
    "Strathcona",
    "Sunset",
    "Victoria-Fraserview",
    "West End",
    "West Point Grey"
]

ZONE_TO_ID = {
    area : id
    for id, area in enumerate(LOCAL_AREAS)
}

ID_TO_ZONE = {
    id : area 
    for area, id in ZONE_TO_ID.items()
}