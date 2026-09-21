# Architecture

PAVAS combines a MMAudio-derived multimodal diffusion backbone with object-centric physics conditioning.

## Main Components

The model uses:

- a MMAudio-derived generator backbone
- offline CLIP patch features
- PPE-derived mass, velocity, and segmentation signals
- PAVAS-specific conditioning modules that inject object-level physics into generation

## High-Level Data Flow

At a high level:

1. video, text, and synchronization features follow the MMAudio backbone design
2. object masks select patch-level visual tokens for each tracked object
3. mass and velocity are embedded into conditioning features
4. the combined object representation is injected into the diffusion backbone

## Repository Naming

The repository-level name stays `PAVAS`. Phy-Adapter is treated as a method component inside the full system rather than as the repository name.
