import pydrake.math

RotationMatrix_AutoDiffXd = getattr(pydrake.math, [name for name in dir(pydrake.math) 
                                  if "RotationMatrix" in name and "AutoDiff" in name][0])
RigidTransform_AutoDiffXd = getattr(pydrake.math, [name for name in dir(pydrake.math) 
                                 if "RigidTransform" in name and "AutoDiff" in name][0])
print([name for name in dir(pydrake.math) if "RotationMatrix" in name and "AutoDiff" in name])