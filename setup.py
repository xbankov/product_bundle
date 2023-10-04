from setuptools import setup, find_packages

setup(
    name='product_bundle',
    version='0.1',
    description='Recommend product bundles using E-commerce dataset',
    author='Mikuláš Bankovič',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)