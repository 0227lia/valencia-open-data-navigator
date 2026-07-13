from src.catalog import normalize_text, tokenize, validate_packages


def test_normalize_text_folds_spanish_and_valencian_accents() -> None:
    assert normalize_text("Valencia: bicicletes, contaminacio!") == "valencia bicicletes contaminacio"
    assert tokenize("Datos de bicicletas y aire") == ["datos", "bicicletas", "aire"]


def test_validation_detects_duplicate_dataset_names() -> None:
    validation = validate_packages(
        [
            {"name": "mobility", "title": "Mobility", "resources": []},
            {"name": "mobility", "title": "Mobility copy", "resources": []},
        ]
    )
    assert validation.duplicate_names == ("mobility",)
    assert not validation.is_valid
