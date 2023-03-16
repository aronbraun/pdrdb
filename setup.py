import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pdrdb',
    version='1.0.0',
    author='Aron Podrigal',
    author_email='a@arontel.com',
    description='revers sqlalchemy orm from postgres sql & advanced asyncpg driver',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/gizber-donations/pdrdb',
    license='MIT',
    packages=['pdrdb'],
    # install_requires=['will-see'],
)