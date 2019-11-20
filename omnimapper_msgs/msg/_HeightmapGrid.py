# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from omnimapper_msgs/HeightmapGrid.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import geometry_msgs.msg
import nav_msgs.msg
import genpy
import std_msgs.msg

class HeightmapGrid(genpy.Message):
  _md5sum = "d58d3b244fb94f14e8b69645aead8102"
  _type = "omnimapper_msgs/HeightmapGrid"
  _has_header = True #flag to mark the presence of a Header object
  _full_text = """std_msgs/Header header
nav_msgs/MapMetaData info
float64 min_height
float64 max_height
int8 min_value
int8 max_value
int8[] data
int8[] nx
int8[] ny
int8[] nz

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
MSG: nav_msgs/MapMetaData
# This hold basic information about the characterists of the OccupancyGrid

# The time at which the map was loaded
time map_load_time
# The map resolution [m/cell]
float32 resolution
# Map width [cells]
uint32 width
# Map height [cells]
uint32 height
# The origin of the map [m, m, rad].  This is the real-world pose of the
# cell (0,0) in the map.
geometry_msgs/Pose origin
================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
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
"""
  __slots__ = ['header','info','min_height','max_height','min_value','max_value','data','nx','ny','nz']
  _slot_types = ['std_msgs/Header','nav_msgs/MapMetaData','float64','float64','int8','int8','int8[]','int8[]','int8[]','int8[]']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       header,info,min_height,max_height,min_value,max_value,data,nx,ny,nz

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(HeightmapGrid, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.info is None:
        self.info = nav_msgs.msg.MapMetaData()
      if self.min_height is None:
        self.min_height = 0.
      if self.max_height is None:
        self.max_height = 0.
      if self.min_value is None:
        self.min_value = 0
      if self.max_value is None:
        self.max_value = 0
      if self.data is None:
        self.data = []
      if self.nx is None:
        self.nx = []
      if self.ny is None:
        self.ny = []
      if self.nz is None:
        self.nz = []
    else:
      self.header = std_msgs.msg.Header()
      self.info = nav_msgs.msg.MapMetaData()
      self.min_height = 0.
      self.max_height = 0.
      self.min_value = 0
      self.max_value = 0
      self.data = []
      self.nx = []
      self.ny = []
      self.nz = []

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
      buff.write(_get_struct_3I().pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_2If2I9d2b().pack(_x.info.map_load_time.secs, _x.info.map_load_time.nsecs, _x.info.resolution, _x.info.width, _x.info.height, _x.info.origin.position.x, _x.info.origin.position.y, _x.info.origin.position.z, _x.info.origin.orientation.x, _x.info.origin.orientation.y, _x.info.origin.orientation.z, _x.info.origin.orientation.w, _x.min_height, _x.max_height, _x.min_value, _x.max_value))
      length = len(self.data)
      buff.write(_struct_I.pack(length))
      pattern = '<%sb'%length
      buff.write(struct.pack(pattern, *self.data))
      length = len(self.nx)
      buff.write(_struct_I.pack(length))
      pattern = '<%sb'%length
      buff.write(struct.pack(pattern, *self.nx))
      length = len(self.ny)
      buff.write(_struct_I.pack(length))
      pattern = '<%sb'%length
      buff.write(struct.pack(pattern, *self.ny))
      length = len(self.nz)
      buff.write(_struct_I.pack(length))
      pattern = '<%sb'%length
      buff.write(struct.pack(pattern, *self.nz))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.info is None:
        self.info = nav_msgs.msg.MapMetaData()
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 94
      (_x.info.map_load_time.secs, _x.info.map_load_time.nsecs, _x.info.resolution, _x.info.width, _x.info.height, _x.info.origin.position.x, _x.info.origin.position.y, _x.info.origin.position.z, _x.info.origin.orientation.x, _x.info.origin.orientation.y, _x.info.origin.orientation.z, _x.info.origin.orientation.w, _x.min_height, _x.max_height, _x.min_value, _x.max_value,) = _get_struct_2If2I9d2b().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sb'%length
      start = end
      end += struct.calcsize(pattern)
      self.data = struct.unpack(pattern, str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sb'%length
      start = end
      end += struct.calcsize(pattern)
      self.nx = struct.unpack(pattern, str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sb'%length
      start = end
      end += struct.calcsize(pattern)
      self.ny = struct.unpack(pattern, str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sb'%length
      start = end
      end += struct.calcsize(pattern)
      self.nz = struct.unpack(pattern, str[start:end])
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
      buff.write(_get_struct_3I().pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_2If2I9d2b().pack(_x.info.map_load_time.secs, _x.info.map_load_time.nsecs, _x.info.resolution, _x.info.width, _x.info.height, _x.info.origin.position.x, _x.info.origin.position.y, _x.info.origin.position.z, _x.info.origin.orientation.x, _x.info.origin.orientation.y, _x.info.origin.orientation.z, _x.info.origin.orientation.w, _x.min_height, _x.max_height, _x.min_value, _x.max_value))
      length = len(self.data)
      buff.write(_struct_I.pack(length))
      pattern = '<%sb'%length
      buff.write(self.data.tostring())
      length = len(self.nx)
      buff.write(_struct_I.pack(length))
      pattern = '<%sb'%length
      buff.write(self.nx.tostring())
      length = len(self.ny)
      buff.write(_struct_I.pack(length))
      pattern = '<%sb'%length
      buff.write(self.ny.tostring())
      length = len(self.nz)
      buff.write(_struct_I.pack(length))
      pattern = '<%sb'%length
      buff.write(self.nz.tostring())
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.info is None:
        self.info = nav_msgs.msg.MapMetaData()
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 94
      (_x.info.map_load_time.secs, _x.info.map_load_time.nsecs, _x.info.resolution, _x.info.width, _x.info.height, _x.info.origin.position.x, _x.info.origin.position.y, _x.info.origin.position.z, _x.info.origin.orientation.x, _x.info.origin.orientation.y, _x.info.origin.orientation.z, _x.info.origin.orientation.w, _x.min_height, _x.max_height, _x.min_value, _x.max_value,) = _get_struct_2If2I9d2b().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sb'%length
      start = end
      end += struct.calcsize(pattern)
      self.data = numpy.frombuffer(str[start:end], dtype=numpy.int8, count=length)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sb'%length
      start = end
      end += struct.calcsize(pattern)
      self.nx = numpy.frombuffer(str[start:end], dtype=numpy.int8, count=length)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sb'%length
      start = end
      end += struct.calcsize(pattern)
      self.ny = numpy.frombuffer(str[start:end], dtype=numpy.int8, count=length)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sb'%length
      start = end
      end += struct.calcsize(pattern)
      self.nz = numpy.frombuffer(str[start:end], dtype=numpy.int8, count=length)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_3I = None
def _get_struct_3I():
    global _struct_3I
    if _struct_3I is None:
        _struct_3I = struct.Struct("<3I")
    return _struct_3I
_struct_2If2I9d2b = None
def _get_struct_2If2I9d2b():
    global _struct_2If2I9d2b
    if _struct_2If2I9d2b is None:
        _struct_2If2I9d2b = struct.Struct("<2If2I9d2b")
    return _struct_2If2I9d2b
