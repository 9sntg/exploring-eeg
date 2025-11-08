def discover_dataset_root(possible_roots=None):
    """
    Try to locate the EEG dataset root directory on PSC.
    """
    if possible_roots is None:
        possible_roots = [
            "/ocean/projects/cis250019p/gandotra/11785-gp-eeg",
            "/ocean/projects/cis240001p/shared/eeg-dataset",
            "./11785-gp-eeg"
        ]

    for root in possible_roots:
        if os.path.exists(os.path.join(root, "ds005589")):
            print(f"✅ Found EEG dataset at: {root}")
            return root

    print("⚠️ EEG dataset not found. You may need to mount or copy it.")
    return None

/ocean/projects/cis250019p/gandotra/11785-gp-eeg/captions.txt
# Locate EEG data
