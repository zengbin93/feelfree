# coding: utf-8
from setuptools import setup, find_packages
import feelfree

setup(
    name="feelfree",
    version=feelfree.version,
    keywords=("对话机器人", "检索式对话系统"),
    description="检索式框架，预期做成 python 版的 AnyQ",
    long_description="对话系统开发的终极目标：Feel free to ask me anything",
    license="MIT",

    url="https://github.com/zengbin93/feelfree",
    author=chan.author,
    author_email=chan.email,

    packages=find_packages(exclude=['test', 'images']),
    include_package_data=True,
    install_requires=[
        "requests", "pandas", "tushare"
    ],

    classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ]
)
