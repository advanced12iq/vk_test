Open Set Logo Detection Dataset (OSLD Dataset)

This is a dataset of product and logo images, released for open set logo detection research
under creative commons (CC BY-NC 4.0) license (please see LICENSE file for details).

Available for download at: https://github.com/mubastan/osld

Contact: Muhammet Bastan (mubastan@gmail.com)

Reference:
@article{OSLD,
  author    = {Muhammet Bastan and
               Hao-Yu Wu and
               Tian Cao and
               Bhargava Kota and
               Mehmet Tek},
  title     = {{Large Scale Open-Set Deep Logo Detection}},
  year      = {2019},
  url       = {http://arxiv.org/abs/1911.07440}
}

Updates:
30 August 2021: Initial release of the dataset

------------------------------------------------------------------------------------------------------------------------

DATASET DESCRIPTION
-------------------
The dataset contains:
    * 20K product images, with logo bounding box annotations
    * 12.1K logo classes with 20.8K canonical logo images downloaded from the Web
    * Product image logo bounding box to canonical logo image match pair annotations

The canonical logo images were downloaded from the web by MTurk annotators, by performing such queries as "brand_name logo"
on web search engines and dowloading the relevant images.
Therefore, the canonical logo images are not necessarily official logo images, but web images that can help recognize
logos in product images. See the above paper for sample images.

FILES AND FORMATS
-----------------

The 20K product images are split into training (train), validation (val) and test sets.

Product images: product-images.tar.gz contains the 20K training (train*.jpg), validation (val*.jpg) and test (test*.jpg) images.

Canonical logo images: logo-images.tar.gz contains the 20.8K canonical logo images for 12.1K logo classes. The files
are named by the logo class names as in amazon-1.jpg, amazon-2.jpg,..., amazon-5.jpg, etc. The image IDs start from 1,
and not necessarily consecutive (this is because duplicate images were deleted, and images were not renamed - TODO).
Space " " character is replaced with "_", e.g., "western_digital-1.jpg".

Annotations: annotations.tar.gz contains the bounding box, matching pair annotations for train/val/test splits.

Bounding box and matching pair annotations for train/val/test splits provided as both pickle (Python 3) and JSON files.
Pickle files: osld-train.pkl, osld-val.pkl, osld-test.pkl
JSON files: osld-train.json, osld-val.json, osld-test.json
The annotations are stored as dictionaries keyed on the product image file name, as follows:
dictionary[product_image_filename] = [(bbox1, logo_image_filename1), (bbox2, logo_image_filename2), ...]
Bounding box coordinates: (x1, y1, x2, y2): (left, top, right, bottom)

Example: "val00033.jpg": [([172, 244, 280, 327], "bell-3.jpg")]

This means image "val00033.jpg" has a logo region with bounding box [172, 244, 280, 327] that matches to the canonical
logo image file "bell-3.jpg".

If there is no logo in the product image, then dictionary[product_image_filename] = [].

If logo_image_filename is "__unknown__", this logo bounding box is not labeled. These unlabeled bounding boxes are used in
training/validation/test of the logo localizer, but cannot be used in the matching.
Note: If you can label these "__unknown__" bounding boxes and share the labels with us, we will be happy to update
the dataset and cite your contribution.

