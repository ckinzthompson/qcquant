import io
import os
import re
import setuptools


NAME = "qcquant"
META_PATH = os.path.join("qcquant","__init__.py")
KEYWORDS = ["gui", "science","chemistry","physics","biology"]
CLASSIFIERS = [
	"Programming Language :: Python :: 3",
	"License :: OSI Approved :: License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
	"Operating System :: OS Independent",
	"Topic :: Scientific/Engineering :: Biology",
	"Topic :: Scientific/Engineering :: Chemistry",
	"Topic :: Scientific/Engineering :: Physics",
	"Environment :: X11 Applications :: Qt",
]
INSTALL_REQUIRES = [
	"numpy>=1.24.0",
	"numba",
	"tifffile",
	"scipy",
	"matplotlib>=3.7.0",
	"napari>=0.4.17",
	"PyQt5",
	"emcee",
	"corner"
]
EXTRAS_REQUIRE = {
	"docs": [
	],
	"tests": [
	],
	"release": [
	],
}
EXTRAS_REQUIRE["dev"] = EXTRAS_REQUIRE["tests"] + EXTRAS_REQUIRE["docs"]

###############################################################################

HERE = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
	"""
	Build an absolute path from *parts* and and return the contents of the
	resulting file.  Assume UTF-8 encoding.
	"""
	with io.open(os.path.join(HERE, *parts), encoding="utf-8") as f:
		return f.read()

META_FILE = read(META_PATH)

def find_meta(meta):
	"""
	Extract __*meta*__ from META_FILE.
	"""
	meta_match = re.search(
		r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
		META_FILE, re.M
	)
	if meta_match:
		return meta_match.group(1)
	raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
	setuptools.setup(
		name=NAME,
		description=find_meta("description"),
		license=find_meta("license"),
		url=find_meta("url"),
		version=find_meta("version"),
		author=find_meta("author"),
		keywords=KEYWORDS,
		long_description=read("README.md"),
		long_description_content_type='text/markdown',
		packages=setuptools.find_packages(where="."),
		package_dir={"": "."},
		package_data={
			"": ["*.png","*.svg"],
		},
		zip_safe=False,
		classifiers=CLASSIFIERS,
		install_requires=INSTALL_REQUIRES,
		extras_require=EXTRAS_REQUIRE,
		entry_points={
				'console_scripts': [
					'qcquant=qcquant.bin.launch_qcquant:main',
				],
		},
	)
