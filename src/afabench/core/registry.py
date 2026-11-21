# from afabench.core.types import (
#     AFAClassifier,
#     AFADataset,
#     AFAInitializer,
#     AFAMethod,
#     AFAUnmasker,
# )

REGISTERED_CLASSES = {
    "L2MTabICLAFAMethod": "afabench.components.methods.discriminative.l2m_tabicl.afa_methods.L2MTabICLAFAMethod",
    "L2MTabICLBackboneAFAMethod": "afabench.components.methods.discriminative.l2m_tabicl_backbone.afa_methods.L2MTabICLBackboneAFAMethod",
    # AFA Method Classes
    "L2MAFAMethod": "afabench.components.methods.discriminative.l2m.afa_methods.L2MAFAMethod",
    "RLAFAMethod": "afabench.components.methods.rl.common.afa_methods.RLAFAMethod",
    "GDFSAFAMethod": "afabench.components.methods.discriminative.gdfs.afa_methods.GDFSAFAMethod",
    "DIMEAFAMethod": "afabench.components.methods.discriminative.dime.afa_methods.DIMEAFAMethod",
    "EDDIAFAMethod": "afabench.components.methods.generative.eddi.afa_methods.EDDIAFAMethod",
    "AACOAFAMethod": "afabench.components.methods.oracle.aaco.afa_methods.AACOAFAMethod",
    "AACONNAFAMethod": "afabench.components.methods.oracle.aaco.nn.AACONNAFAMethod",
    "StaticBaseMethod": "afabench.components.methods.static.pt.StaticBaseMethod",
    "RandomWithoutClassifierAFAMethod": "afabench.components.methods.dummy.without_classifier.RandomWithoutClassifierAFAMethod",
    "SequentialWithoutClassifierAFAMethod": "afabench.components.methods.dummy.without_classifier.SequentialWithoutClassifierAFAMethod",
    "RandomWithClassifierAFAMethod": "afabench.components.methods.dummy.with_classifier.RandomWithClassifierAFAMethod",
    "SequentialWithClassifierAFAMethod": "afabench.components.methods.dummy.with_classifier.SequentialWithClassifierAFAMethod",
    # Legacy aliases for backward compatibility with saved bundles
    "RandomDummyAFAMethod": "afabench.components.methods.dummy.without_classifier.RandomWithoutClassifierAFAMethod",
    "SequentialDummyAFAMethod": "afabench.components.methods.dummy.without_classifier.SequentialWithoutClassifierAFAMethod",
    "RandomSelectionAFAMethod": "afabench.components.methods.dummy.with_classifier.RandomWithClassifierAFAMethod",
    "SequentialSelectionAFAMethod": "afabench.components.methods.dummy.with_classifier.SequentialWithClassifierAFAMethod",
    # AFA Dataset Classes
    "CubeDataset": "afabench.datasets.cube.CubeDataset",
    "CubeNonUniformCostsDataset": "afabench.datasets.cube_nonuniform_costs.CubeNonUniformCostsDataset",
    "CubeNMDataset": "afabench.datasets.cube_nm.CubeNMDataset",
    "AFAContextDataset": "afabench.datasets.cube_nm.CubeNMDataset",  # legacy alias to CubeNM
    "MNISTDataset": "afabench.datasets.mnist.MNISTDataset",
    "SyntheticMNISTDataset": "afabench.datasets.synthetic_mnist.SyntheticMNISTDataset",
    "DiabetesDataset": "afabench.datasets.diabetes.DiabetesDataset",
    "PhysionetChallenge2012Dataset": (
        "afabench.datasets.physionet_challenge_2012.PhysionetChallenge2012Dataset"
    ),
    "MiniBooNEDataset": "afabench.datasets.miniboone.MiniBooNEDataset",
    "MiniBooNESubsetDataset": "afabench.datasets.miniboone_subset.MiniBooNESubsetDataset",
    "FashionMNISTDataset": "afabench.datasets.fashion_mnist.FashionMNISTDataset",
    "BankMarketingDataset": "afabench.datasets.bank_marketing.BankMarketingDataset",
    "CKDDataset": "afabench.datasets.ckd.CKDDataset",
    "ACTG175Dataset": "afabench.datasets.actg.ACTG175Dataset",
    "ImagenetteDataset": "afabench.datasets.imagenette.ImagenetteDataset",
    "MetabricDataset": "afabench.datasets.metabric.MetabricDataset",
    # AFA Classifier Classes
    "RandomDummyAFAClassifier": "afabench.components.classifiers.RandomDummyAFAClassifier",
    "UniformDummyAFAClassifier": "afabench.components.classifiers.UniformDummyAFAClassifier",
    "WrappedMaskedMLPClassifier": "afabench.components.classifiers.WrappedMaskedMLPClassifier",
    "WrappedMaskedViTClassifier": "afabench.components.classifiers.WrappedMaskedViTClassifier",
    "JAFAAFAClassifier": "afabench.components.methods.rl.jafa.models.JAFAAFAClassifier",
    "ODINAFAClassifier": "afabench.components.methods.rl.odin.models.ODINAFAClassifier",
    "OLAFAClassifier": "afabench.components.methods.rl.ol.models.OLAFAClassifier",
    "GreedyAFAClassifier": "afabench.components.methods.discriminative.common.models.GreedyAFAClassifier",
    # AFA Unmasker Classes
    "DirectUnmasker": "afabench.components.unmaskers.direct_unmasker.DirectUnmasker",
    "ImagePatchUnmasker": "afabench.components.unmaskers.image_patch_unmasker.ImagePatchUnmasker",
    "CubeNMUnmasker": "afabench.components.unmaskers.cube_nm_unmasker.CubeNMUnmasker",
    # AFA Initializer Classes
    "ZeroInitializer": "afabench.components.initializers.zero_initializer.ZeroInitializer",
    "FixedRandomInitializer": "afabench.components.initializers.fixed_random_initializer.FixedRandomInitializer",
    "ManualInitializer": "afabench.components.initializers.manual_initializer.ManualInitializer",
    "MutualInformationInitializer": "afabench.components.initializers.mutual_information_initializer.MutualInformationInitializer",
    "LeastInformativeInitializer": "afabench.components.initializers.least_informative_initializer.LeastInformativeInitializer",
    "RandomInitializer": "afabench.components.initializers.random_initializer.RandomInitializer",
    # In-context pretrained models
    "L2MModel": "afabench.components.methods.discriminative.l2m.models.L2MModel",
    # General PyTorch Model Bundle
    "TorchModelBundle": "afabench.core.bundle_system.torch_bundle.TorchModelBundle",
}


def get_class(class_name: str) -> type:
    """Return a reference to a registered class, given the class name."""
    if class_name not in REGISTERED_CLASSES:
        msg = f"Unknown class: {class_name}"
        raise ValueError(msg)

    module_name, class_name_in_module = REGISTERED_CLASSES[class_name].rsplit(
        ".", 1
    )
    module = __import__(module_name, fromlist=[class_name_in_module])
    return getattr(module, class_name_in_module)


# Backward compatibility functions - these delegate to the unified get_class function
# def get_afa_method_class(class_name: str) -> type[AFAMethod]:
#     """Return a reference to an AFAMethod class, given the class name."""
#     return get_class(class_name)


# def get_afa_dataset_class(class_name: str) -> type[AFADataset]:
#     """Return a reference to an AFADataset class, given the class name."""
#     return get_class(class_name)


# def get_afa_classifier_class(class_name: str) -> type[AFAClassifier]:
#     """Return a reference to an AFAClassifier class, given the class name."""
#     return get_class(class_name)


# def get_afa_unmasker_class(class_name: str) -> type[AFAUnmasker]:
#     """Return a reference to an AFAUnmasker class, given the class name."""
#     return get_class(class_name)


# def get_afa_initializer_class(class_name: str) -> type[AFAInitializer]:
#     """Return a reference to an AFAInitializer class, given the class name."""
#     return get_class(class_name)
