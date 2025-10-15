"""
Fish taxa definitions for re-use across scripts.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FishTaxon:
    """
    Structured fish taxon entry.

    Attributes:
        binomial: Scientific name.
        commonName: Common name.
        taxClass: Taxonomic class.
        taxOrder: Taxonomic order.
        taxFamily: Taxonomic family.
        genus: Genus name.
    """

    binomial: str
    commonName: str
    taxClass: str
    taxOrder: str
    taxFamily: str
    genus: str


taxaTuples: List[Tuple[str, str, str, str, str, str]] = [
    ("Salmo salar", "Atlantic salmon", "Actinopterygii", "Salmoniformes", "Salmonidae", "Salmo"),
    ("Oncorhynchus mykiss", "Rainbow trout", "Actinopterygii", "Salmoniformes", "Salmonidae", "Oncorhynchus"),
    ("Gadus morhua", "Atlantic cod", "Actinopterygii", "Gadiformes", "Gadidae", "Gadus"),
    ("Melanogrammus aeglefinus", "Haddock", "Actinopterygii", "Gadiformes", "Gadidae", "Melanogrammus"),
    ("Scomber scombrus", "Atlantic mackerel", "Actinopterygii", "Scombriformes", "Scombridae", "Scomber"),
    ("Thunnus thynnus", "Atlantic bluefin tuna", "Actinopterygii", "Scombriformes", "Scombridae", "Thunnus"),
    ("Coryphaena hippurus", "Mahi-mahi", "Actinopterygii", "Carangiformes", "Coryphaenidae", "Coryphaena"),
    ("Xiphias gladius", "Swordfish", "Actinopterygii", "Xiphiiformes", "Xiphiidae", "Xiphias"),
    ("Clupea harengus", "Atlantic herring", "Actinopterygii", "Clupeiformes", "Clupeidae", "Clupea"),
    ("Sardina pilchardus", "European pilchard", "Actinopterygii", "Clupeiformes", "Clupeidae", "Sardina"),
    ("Engraulis encrasicolus", "European anchovy", "Actinopterygii", "Clupeiformes", "Engraulidae", "Engraulis"),
    ("Amphiprion ocellaris", "Ocellaris clownfish", "Actinopterygii", "Blenniiformes", "Pomacentridae", "Amphiprion"),
    ("Pomacanthus imperator", "Emperor angelfish", "Actinopterygii", "Acanthuriformes", "Pomacanthidae", "Pomacanthus"),
    ("Pterois volitans", "Red lionfish", "Actinopterygii", "Scorpaeniformes", "Scorpaenidae", "Pterois"),
    ("Zebrasoma flavescens", "Yellow tang", "Actinopterygii", "Acanthuriformes", "Acanthuridae", "Zebrasoma"),
    ("Hippocampus kuda", "Common seahorse", "Actinopterygii", "Syngnathiformes", "Syngnathidae", "Hippocampus"),
    ("Betta splendens", "Siamese fighting fish", "Actinopterygii", "Anabantiformes", "Osphronemidae", "Betta"),
    ("Paracheirodon innesi", "Neon tetra", "Actinopterygii", "Characiformes", "Characidae", "Paracheirodon"),
    ("Carassius auratus", "Goldfish", "Actinopterygii", "Cypriniformes", "Cyprinidae", "Carassius"),
    ("Cyprinus carpio", "Common carp", "Actinopterygii", "Cypriniformes", "Cyprinidae", "Cyprinus"),
    ("Poecilia reticulata", "Guppy", "Actinopterygii", "Cyprinodontiformes", "Poeciliidae", "Poecilia"),
    ("Astatotilapia burtoni", "Burton’s mouthbrooder", "Actinopterygii", "Cichliformes", "Cichlidae", "Astatotilapia"),
    ("Oreochromis niloticus", "Nile tilapia", "Actinopterygii", "Cichliformes", "Cichlidae", "Oreochromis"),
    ("Pterophyllum scalare", "Freshwater angelfish", "Actinopterygii", "Cichliformes", "Cichlidae", "Pterophyllum"),
    ("Micropterus salmoides", "Florida bass", "Actinopterygii", "Centrarchiformes", "Centrarchidae", "Micropterus"),
    ("Lepomis macrochirus", "Bluegill sunfish", "Actinopterygii", "Centrarchiformes", "Centrarchidae", "Lepomis"),
    ("Esox lucius", "Northern pike", "Actinopterygii", "Esociformes", "Esocidae", "Esox"),
    ("Ictalurus punctatus", "Channel catfish", "Actinopterygii", "Siluriformes", "Ictaluridae", "Ictalurus"),
    ("Silurus glanis", "Wels catfish", "Actinopterygii", "Siluriformes", "Siluridae", "Silurus"),
    ("Electrophorus electricus", "Electric eel", "Actinopterygii", "Gymnotiformes", "Gymnotidae", "Electrophorus"),
    ("Arapaima gigas", "Arapaima", "Actinopterygii", "Osteoglossiformes", "Arapaimidae", "Arapaima"),
    ("Osteoglossum bicirrhosum", "Silver arowana", "Actinopterygii", "Osteoglossiformes", "Osteoglossidae", "Osteoglossum"),
    ("Anguilla anguilla", "European eel", "Actinopterygii", "Anguilliformes", "Anguillidae", "Anguilla"),
    ("Muraena helena", "Mediterranean moray", "Actinopterygii", "Anguilliformes", "Muraenidae", "Muraena"),
    ("Lophius piscatorius", "Monkfish", "Actinopterygii", "Lophiiformes", "Lophiidae", "Lophius"),
    ("Hippoglossus hippoglossus", "Atlantic halibut", "Actinopterygii", "Pleuronectiformes", "Pleuronectidae", "Hippoglossus"),
    ("Pleuronectes platessa", "European plaice", "Actinopterygii", "Pleuronectiformes", "Pleuronectidae", "Pleuronectes"),
    ("Sphyraena barracuda", "Great barracuda", "Actinopterygii", "Carangiformes", "Sphyranidae", "Sphyraena"),
    ("Dicentrarchus labrax", "European seabass", "Actinopterygii", "Acanthuriformes", "Moronidae", "Dicentrarchus"),
    ("Lutjanus campechanus", "Northern red snapper", "Actinopterygii", "Acanthuriformes", "Lutjanidae", "Lutjanus"),
    ("Epinephelus itajara", "Goliath grouper", "Actinopterygii", "Perciformes", "Epinephelidae", "Epinephelus"),
    ("Cheilinus undulatus", "Humphead wrasse", "Actinopterygii", "Labriformes", "Labridae", "Cheilinus"),
    ("Gobius niger", "Black goby", "Actinopterygii", "Gobiiformes", "Gobiidae", "Gobius"),
    ("Carcharodon carcharias", "Great white shark", "Chondrichthyes", "Lamniformes", "Lamnidae", "Carcharodon"),
    ("Galeocerdo cuvier", "Tiger shark", "Chondrichthyes", "Carcharhiniformes", "Carcharhinidae", "Galeocerdo"),
    ("Sphyrna lewini", "Scalloped hammerhead", "Chondrichthyes", "Carcharhiniformes", "Sphyrnidae", "Sphyrna"),
    ("Raja clavata", "Thornback ray", "Chondrichthyes", "Rajiformes", "Rajidae", "Raja"),
    ("Mobula birostris", "Giant manta ray", "Chondrichthyes", "Myliobatiformes", "Mobulidae", "Mobula"),
    ("Takifugu rubripes", "Japanese pufferfish", "Actinopterygii", "Tetraodontiformes", "Tetraodontidae", "Takifugu"),
    ("Diodon hystrix", "Porcupinefish", "Actinopterygii", "Tetraodontiformes", "Diodontidae", "Diodon"),
]


taxa: List[FishTaxon] = [FishTaxon(*t) for t in taxaTuples]
