REINVENT 4.0 input examples
===========================

A set of input files to explain each run mode in REINVENT4.  Please note
that these example are meant for demonstration only and will need to be
adjusted for actual production.


Format
------

The file format is in TOML format, see https://toml.io/en/.


Examples
--------

TOML file              | description
-----------------------|----------------------------------------------------
sampling.toml          | shows how one can sample molecules from a model.
scoring.toml           | shows how to run scoring on a set of compounds.
transfer\_learning.toml | shows how to run transfer learning.
staged\_learning.toml   | shows how to run reinforcement/curriculum learning.
dc\_multi.toml          | staged learning using DeepChemDMPNN scoring.
dc\_multi\_stage1\_scoring.toml | stage 1 scoring file for dc\_multi.toml.
dc\_multi\_stage2\_scoring.toml | stage 2 scoring file for dc\_multi.toml.
