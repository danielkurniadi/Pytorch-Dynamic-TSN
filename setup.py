from setuptools import setup, find_packages

"""
CLI script installation
"""
setup(
    name = 'core',
    version = '0.1',
    packages = find_packages(),
    include_package_data = True,
    install_requires = [
        'Click',
    ],
    entry_points = {
        'console_scripts' : [
            'dataset=core.cli:dataset'
        ]
    },
)
