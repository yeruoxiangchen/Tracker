TRELLIS-500K / ABO two-object inspection subset
================================================

Purpose:
Provide two objects that are explicitly present in the frozen TRELLIS-500K
ABO.csv selection, together with the corresponding ABO real catalog images
that TRELLIS does not use in its standard synthetic render_cond training path.

Created at UTC:
2026-08-19T14:40:20Z

Sources:
TRELLIS frozen metadata:
https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K/resolve/main/ABO.csv

ABO mesh payloads:
https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/

ABO catalog image identity metadata:
https://amazon-berkeley-objects.s3.amazonaws.com/images/metadata/images.csv.gz

ABO real catalog image download template documented by ABO:
https://m.media-amazon.com/images/I/<image_id>.<extension>

Object 1:
item_id: B07YBH2SR3
product: AmazonBasics Solid Wood Kid Chair Set, White, 26.5 inch, 2-Pack
TRELLIS file_identifier: 3/B07YBH2SR3.glb
TRELLIS aesthetic_score: 5.136597633361816
TRELLIS frozen mesh SHA256:
0006d4c69de70f84754df85c2ec0a34514223f941500f080c84c19da7e137998
Note: the catalog listing is a two-chair pack, while the GLB represents one
chair instance. The six images remain the official images bound to that exact
ABO item/model identity.

Object 2:
item_id: B07HSB4WV6
product: Amazon Brand - Rivet Vermont Modern Diamond-Stitched Kitchen Counter
Bar Stool, 42 inch high, Chalk White and Brass
TRELLIS file_identifier: 6/B07HSB4WV6.glb
TRELLIS aesthetic_score: 4.7749128341674805
TRELLIS frozen mesh SHA256:
00128c5f4884f93566f9250fbe2ba439351190d2aa417e64e52b0ae0f2565222

Directory contract per object:
mesh/<item_id>.glb
    Exact ABO GLB selected by TRELLIS-500K. Textures are embedded in GLB.

real_catalog_images/main_<image_id>.jpg
real_catalog_images/other_<index>_<image_id>.jpg
    Original-resolution real Amazon catalog photographs associated with the
    same ABO listing and 3D model. They are not calibrated multiview captures:
    there are no exact camera intrinsics/extrinsics or ground-truth masks.

identity.json
    Frozen mapping between TRELLIS row, ABO listing, GLB and image identities.

Integrity:
Run sha256sum -c SHA256SUMS.txt from this directory.

License and attribution:
ABO data is CC BY 4.0. Credit for the images and 3D models belongs to
Amazon.com. See LICENSE-CC-BY-4.0.txt.
