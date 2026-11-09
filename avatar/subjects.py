"""Subject, camera and sequence identifiers of the datasets used in the paper."""

# NeRSemble subject ids. The first ``--num_train`` entries are used for training; the remaining ones
# are held out for test-time inversion.
NERSEMBLE_SUBJECTS = [
    17, 18, 24, 30, 31, 32, 33, 35, 36, 37, 38, 40, 42, 43, 55, 56, 57, 59, 61, 63, 64, 67, 69, 72,
    73, 74, 75, 81, 82, 83, 85, 88, 89, 90, 92, 93, 96, 97, 99, 100, 104, 105, 106, 107, 108, 109,
    111, 112, 115, 116, 117, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
    135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 151, 162, 164, 165, 166, 167,
    170, 171, 172, 173, 174, 175, 176, 177, 178, 182, 183, 184, 185, 186, 189, 190, 191, 192, 193,
    194, 196, 197, 198, 199, 200, 201, 202, 209, 210, 211, 212, 214, 215, 216, 217, 218, 220, 221,
    222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 236, 237, 238, 239, 240, 241, 244, 245, 246,
    247, 248, 249, 250, 251, 252, 253, 254, 256, 257, 258, 259, 260, 261, 262, 263, 264, 267, 269,
    270, 271, 272, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 293,
    294, 298, 299, 301, 302, 304, 305, 306, 307, 308, 309, 310, 311, 312, 314, 315, 316, 317, 318,
    319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 330, 331, 332, 333, 368, 369, 370, 371, 372,
    373, 374, 375,
]

# NeRSemble camera serials. A view index ``i`` in [0, 16) refers to ``NERSEMBLE_CAMERAS[i]``.
NERSEMBLE_CAMERAS = [
    "222200042", "222200044", "222200046", "222200040", "222200036", "222200048", "220700191", "222200041",
    "222200037", "222200038", "222200047", "222200043", "222200049", "222200039", "222200045", "221501007",
]

# NeRSemble sequences used for training. The neutral reference mesh of a subject is frame 0 of
# ``EXP-5-mouth`` (index 3).
NERSEMBLE_SEQUENCES = ["EXP-3-cheeks+nose", "EXP-4-lips", "EXP-2-eyes", "EXP-5-mouth", "EMO-1-shout+laugh",
                       "EMO-2-surprise+fear"]
NEUTRAL_SEQUENCE_INDEX = 3

# Monocular in-the-wild videos tracked with VHAP.
ITW_SUBJECTS = ["bala", "biden", "obama", "malte_1", "wojtek_1", "justin", "nf_02", "person_0000", "person_0004"]
