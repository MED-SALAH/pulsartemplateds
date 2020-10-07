from setuptools import find_packages,setup


setup(
    name="{{cookiecutter.project_name}}",
    version="0.1",
    packages=find_packages(),
    description="lib for DS",
    author="Big Apps",
    author_email="zachour@bigapps.fr",
    license='?',
    install_requires=[],
    tests_require=['pytest'],
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
    ],
)
