# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from omnimapper_msgs/OrientationCorrection.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import geometry_msgs.msg
import genpy

class OrientationCorrection(genpy.Message):
  _md5sum = "faf32340078ffd57016e879e585d1b93"
  _type = "omnimapper_msgs/OrientationCorrection"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """time from_stamp
time to_stamp
geometry_msgs/Quaternion correction
geometry_msgs/Vector3 drift

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z"""
  __slots__ = ['from_stamp','to_stamp','correction','drift']
  _slot_types = ['time','time','geometry_msgs/Quaternion','geometry_msgs/Vector3']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       from_stamp,to_stamp,correction,drift

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(OrientationCorrection, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.from_stamp is None:
        self.from_stamp = genpy.Time()
      if self.to_stamp is None:
        self.to_stamp = genpy.Time()
      if self.correction is None:
        self.correction = geometry_msgs.msg.Quaternion()
      if self.drift is None:
        self.drift = geometry_msgs.msg.Vector3()
    else:
      self.from_stamp = genpy.Time()
      self.to_stamp = genpy.Time()
      self.correction = geometry_msgs.msg.Quaternion()
      self.drift = geometry_msgs.msg.Vector3()

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_4I7d().pack(_x.from_stamp.secs, _x.from_stamp.nsecs, _x.to_stamp.secs, _x.to_stamp.nsecs, _x.correction.x, _x.correction.y, _x.correction.z, _x.correction.w, _x.drift.x, _x.drift.y, _x.drift.z))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.from_stamp is None:
        self.from_stamp = genpy.Time()
      if self.to_stamp is None:
        self.to_stamp = genpy.Time()
      if self.correction is None:
        self.correction = geometry_msgs.msg.Quaternion()
      if self.drift is None:
        self.drift = geometry_msgs.msg.Vector3()
      end = 0
      _x = self
      start = end
      end += 72
      (_x.from_stamp.secs, _x.from_stamp.nsecs, _x.to_stamp.secs, _x.to_stamp.nsecs, _x.correction.x, _x.correction.y, _x.correction.z, _x.correction.w, _x.drift.x, _x.drift.y, _x.drift.z,) = _get_struct_4I7d().unpack(str[start:end])
      self.from_stamp.canon()
      self.to_stamp.canon()
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_4I7d().pack(_x.from_stamp.secs, _x.from_stamp.nsecs, _x.to_stamp.secs, _x.to_stamp.nsecs, _x.correction.x, _x.correction.y, _x.correction.z, _x.correction.w, _x.drift.x, _x.drift.y, _x.drift.z))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.from_stamp is None:
        self.from_stamp = genpy.Time()
      if self.to_stamp is None:
        self.to_stamp = genpy.Time()
      if self.correction is None:
        self.correction = geometry_msgs.msg.Quaternion()
      if self.drift is None:
        self.drift = geometry_msgs.msg.Vector3()
      end = 0
      _x = self
      start = end
      end += 72
      (_x.from_stamp.secs, _x.from_stamp.nsecs, _x.to_stamp.secs, _x.to_stamp.nsecs, _x.correction.x, _x.correction.y, _x.correction.z, _x.correction.w, _x.drift.x, _x.drift.y, _x.drift.z,) = _get_struct_4I7d().unpack(str[start:end])
      self.from_stamp.canon()
      self.to_stamp.canon()
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_4I7d = None
def _get_struct_4I7d():
    global _struct_4I7d
    if _struct_4I7d is None:
        _struct_4I7d = struct.Struct("<4I7d")
    return _struct_4I7d
