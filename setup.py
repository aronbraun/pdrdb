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
    packages=setuptools.find_packages(),
    install_requires=[
        "asyncpg==0.27.0",
        "fastapi==0.94.1",
        "phonenumbers==8.13.7",
        "psycopg2==2.9.5",
        "setuptools==65.6.3",
        "sqlacodegen==3.0.0b2",
        "SQLAlchemy==1.4.26",
        "starlette==0.26.1",
    ],
)
