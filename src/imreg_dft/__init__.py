# We may import this during setup invocation
# because of the version we have to query
# However, i.e. numpy may not be installed at setup install time.
try:
    from imreg_dft.imreg import (
        imshow,
        similarity,
        transform_img,
        transform_img_dict,
        translation,
    )
except ImportError as exc:
    print(f"Unable to import the main package: {exc}")


__version__ = "2.0.1a"
