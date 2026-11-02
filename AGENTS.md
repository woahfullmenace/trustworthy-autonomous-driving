# Coursework project instructions

Read PROJECT_PROPOSAL.md before implementing project behavior.

Build a coursework-sized prototype: road image -> five concepts -> shallow decision tree -> STOP / SLOW / GO.
Concepts: pedestrian_present, vehicle_present, cyclist_present, road_clear, obstacle_close.
Include concept intervention, validation-tuned abstention (REQUEST HUMAN CONTROL), corruption evaluation, and an end-to-end ResNet18 baseline in later milestones.

Use the existing local Python 3.14 virtual environment. This computer uses CPU execution; NVIDIA CUDA is optional. Run commands from the repository root with `.venv/Scripts/python.exe`.
Keep source datasets, environments, and model artifacts out of Git. Preserve source archives.
Do not add CARLA, LiDAR, reinforcement learning, depth networks, new detectors, or real vehicle control.
Keep test data out of threshold selection and model tuning. Document proxy labels and their limitations.
Current milestone: concept/action label definition, label inspection, and experimental train/validation/test splits. Model training remains a later milestone.
