# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from omnimapper_msgs/FrameNotificationSymbols.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import omnimapper_msgs.msg
import geometry_msgs.msg
import std_msgs.msg

class FrameNotificationSymbols(genpy.Message):
  _md5sum = "d04202582a4c07f179b01903511f7769"
  _type = "omnimapper_msgs/FrameNotificationSymbols"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """string platform
omnimapper_msgs/FrameNotificationSymbol[] frames
================================================================================
MSG: omnimapper_msgs/FrameNotificationSymbol
Header header
omnimapper_msgs/FrameNotification frame
omnimapper_msgs/GtsamSymbol symbol

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id

================================================================================
MSG: omnimapper_msgs/FrameNotification
# frame types
uint8 POINT_CLOUD=0
uint8 IMAGE=1
uint8 GPS=2
uint8 RAW_GPS=3

#mode types
uint8 SE3=0 #2D mode (ground phase)
uint8 SE2=1 #3D mode (air phase)

# header for time/frame information
#  time: timestamp of the data as it came to omnicache
#  frame: frame that data originally came in
Header header
string platform
# type of data
int32 type
# mode of operation
int32 mode
# Pose information (maybe just for foreign subgraphs?)
geometry_msgs/TransformStamped pose

================================================================================
MSG: geometry_msgs/TransformStamped
# This expresses a transform from coordinate frame header.frame_id
# to the coordinate frame child_frame_id
#
# This message is mostly used by the 
# <a href="http://wiki.ros.org/tf">tf</a> package. 
# See its documentation for more information.

Header header
string child_frame_id # the frame id of the child frame
Transform transform

================================================================================
MSG: geometry_msgs/Transform
# This represents the transform between two coordinate frames in free space.

Vector3 translation
Quaternion rotation

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
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: omnimapper_msgs/GtsamSymbol
uint8 symbol
uint64 index
"""
  __slots__ = ['platform','frames']
  _slot_types = ['string','omnimapper_msgs/FrameNotificationSymbol[]']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       platform,frames

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(FrameNotificationSymbols, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.platform is None:
        self.platform = ''
      if self.frames is None:
        self.frames = []
    else:
      self.platform = ''
      self.frames = []

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
      _x = self.platform
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      length = len(self.frames)
      buff.write(_struct_I.pack(length))
      for val1 in self.frames:
        _v1 = val1.header
        buff.write(_get_struct_I().pack(_v1.seq))
        _v2 = _v1.stamp
        _x = _v2
        buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
        _x = _v1.frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _v3 = val1.frame
        _v4 = _v3.header
        buff.write(_get_struct_I().pack(_v4.seq))
        _v5 = _v4.stamp
        _x = _v5
        buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
        _x = _v4.frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = _v3.platform
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = _v3
        buff.write(_get_struct_2i().pack(_x.type, _x.mode))
        _v6 = _v3.pose
        _v7 = _v6.header
        buff.write(_get_struct_I().pack(_v7.seq))
        _v8 = _v7.stamp
        _x = _v8
        buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
        _x = _v7.frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = _v6.child_frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _v9 = _v6.transform
        _v10 = _v9.translation
        _x = _v10
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v11 = _v9.rotation
        _x = _v11
        buff.write(_get_struct_4d().pack(_x.x, _x.y, _x.z, _x.w))
        _v12 = val1.symbol
        _x = _v12
        buff.write(_get_struct_BQ().pack(_x.symbol, _x.index))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.frames is None:
        self.frames = None
      end = 0
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.platform = str[start:end].decode('utf-8')
      else:
        self.platform = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.frames = []
      for i in range(0, length):
        val1 = omnimapper_msgs.msg.FrameNotificationSymbol()
        _v13 = val1.header
        start = end
        end += 4
        (_v13.seq,) = _get_struct_I().unpack(str[start:end])
        _v14 = _v13.stamp
        _x = _v14
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v13.frame_id = str[start:end].decode('utf-8')
        else:
          _v13.frame_id = str[start:end]
        _v15 = val1.frame
        _v16 = _v15.header
        start = end
        end += 4
        (_v16.seq,) = _get_struct_I().unpack(str[start:end])
        _v17 = _v16.stamp
        _x = _v17
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v16.frame_id = str[start:end].decode('utf-8')
        else:
          _v16.frame_id = str[start:end]
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v15.platform = str[start:end].decode('utf-8')
        else:
          _v15.platform = str[start:end]
        _x = _v15
        start = end
        end += 8
        (_x.type, _x.mode,) = _get_struct_2i().unpack(str[start:end])
        _v18 = _v15.pose
        _v19 = _v18.header
        start = end
        end += 4
        (_v19.seq,) = _get_struct_I().unpack(str[start:end])
        _v20 = _v19.stamp
        _x = _v20
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v19.frame_id = str[start:end].decode('utf-8')
        else:
          _v19.frame_id = str[start:end]
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v18.child_frame_id = str[start:end].decode('utf-8')
        else:
          _v18.child_frame_id = str[start:end]
        _v21 = _v18.transform
        _v22 = _v21.translation
        _x = _v22
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v23 = _v21.rotation
        _x = _v23
        start = end
        end += 32
        (_x.x, _x.y, _x.z, _x.w,) = _get_struct_4d().unpack(str[start:end])
        _v24 = val1.symbol
        _x = _v24
        start = end
        end += 9
        (_x.symbol, _x.index,) = _get_struct_BQ().unpack(str[start:end])
        self.frames.append(val1)
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
      _x = self.platform
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      length = len(self.frames)
      buff.write(_struct_I.pack(length))
      for val1 in self.frames:
        _v25 = val1.header
        buff.write(_get_struct_I().pack(_v25.seq))
        _v26 = _v25.stamp
        _x = _v26
        buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
        _x = _v25.frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _v27 = val1.frame
        _v28 = _v27.header
        buff.write(_get_struct_I().pack(_v28.seq))
        _v29 = _v28.stamp
        _x = _v29
        buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
        _x = _v28.frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = _v27.platform
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = _v27
        buff.write(_get_struct_2i().pack(_x.type, _x.mode))
        _v30 = _v27.pose
        _v31 = _v30.header
        buff.write(_get_struct_I().pack(_v31.seq))
        _v32 = _v31.stamp
        _x = _v32
        buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
        _x = _v31.frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = _v30.child_frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _v33 = _v30.transform
        _v34 = _v33.translation
        _x = _v34
        buff.write(_get_struct_3d().pack(_x.x, _x.y, _x.z))
        _v35 = _v33.rotation
        _x = _v35
        buff.write(_get_struct_4d().pack(_x.x, _x.y, _x.z, _x.w))
        _v36 = val1.symbol
        _x = _v36
        buff.write(_get_struct_BQ().pack(_x.symbol, _x.index))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.frames is None:
        self.frames = None
      end = 0
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.platform = str[start:end].decode('utf-8')
      else:
        self.platform = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.frames = []
      for i in range(0, length):
        val1 = omnimapper_msgs.msg.FrameNotificationSymbol()
        _v37 = val1.header
        start = end
        end += 4
        (_v37.seq,) = _get_struct_I().unpack(str[start:end])
        _v38 = _v37.stamp
        _x = _v38
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v37.frame_id = str[start:end].decode('utf-8')
        else:
          _v37.frame_id = str[start:end]
        _v39 = val1.frame
        _v40 = _v39.header
        start = end
        end += 4
        (_v40.seq,) = _get_struct_I().unpack(str[start:end])
        _v41 = _v40.stamp
        _x = _v41
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v40.frame_id = str[start:end].decode('utf-8')
        else:
          _v40.frame_id = str[start:end]
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v39.platform = str[start:end].decode('utf-8')
        else:
          _v39.platform = str[start:end]
        _x = _v39
        start = end
        end += 8
        (_x.type, _x.mode,) = _get_struct_2i().unpack(str[start:end])
        _v42 = _v39.pose
        _v43 = _v42.header
        start = end
        end += 4
        (_v43.seq,) = _get_struct_I().unpack(str[start:end])
        _v44 = _v43.stamp
        _x = _v44
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v43.frame_id = str[start:end].decode('utf-8')
        else:
          _v43.frame_id = str[start:end]
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v42.child_frame_id = str[start:end].decode('utf-8')
        else:
          _v42.child_frame_id = str[start:end]
        _v45 = _v42.transform
        _v46 = _v45.translation
        _x = _v46
        start = end
        end += 24
        (_x.x, _x.y, _x.z,) = _get_struct_3d().unpack(str[start:end])
        _v47 = _v45.rotation
        _x = _v47
        start = end
        end += 32
        (_x.x, _x.y, _x.z, _x.w,) = _get_struct_4d().unpack(str[start:end])
        _v48 = val1.symbol
        _x = _v48
        start = end
        end += 9
        (_x.symbol, _x.index,) = _get_struct_BQ().unpack(str[start:end])
        self.frames.append(val1)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_2i = None
def _get_struct_2i():
    global _struct_2i
    if _struct_2i is None:
        _struct_2i = struct.Struct("<2i")
    return _struct_2i
_struct_BQ = None
def _get_struct_BQ():
    global _struct_BQ
    if _struct_BQ is None:
        _struct_BQ = struct.Struct("<BQ")
    return _struct_BQ
_struct_4d = None
def _get_struct_4d():
    global _struct_4d
    if _struct_4d is None:
        _struct_4d = struct.Struct("<4d")
    return _struct_4d
_struct_2I = None
def _get_struct_2I():
    global _struct_2I
    if _struct_2I is None:
        _struct_2I = struct.Struct("<2I")
    return _struct_2I
_struct_3d = None
def _get_struct_3d():
    global _struct_3d
    if _struct_3d is None:
        _struct_3d = struct.Struct("<3d")
    return _struct_3d
