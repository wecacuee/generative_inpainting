# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from omnimapper_msgs/FrameRecord.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import omnimapper_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg
import std_msgs.msg

class FrameRecord(genpy.Message):
  _md5sum = "3008d16f445972e03d5b5a411cd9b9c4"
  _type = "omnimapper_msgs/FrameRecord"
  _has_header = True #flag to mark the presence of a Header object
  _full_text = """Header header
string platform
omnimapper_msgs/FrameNotification frame
omnimapper_msgs/GtsamSymbol symbol
omnimapper_msgs/StampedPointCloud cloud
omnimapper_msgs/Centroid centroid

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

================================================================================
MSG: omnimapper_msgs/StampedPointCloud
# The pose and time stamp of point cloud
geometry_msgs/TransformStamped transform
# The platform source of this point cloud
string platform
# The aggregated point cloud
sensor_msgs/PointCloud2 cloud


================================================================================
MSG: sensor_msgs/PointCloud2
# This message holds a collection of N-dimensional points, which may
# contain additional information such as normals, intensity, etc. The
# point data is stored as a binary blob, its layout described by the
# contents of the "fields" array.

# The point cloud data may be organized 2d (image-like) or 1d
# (unordered). Point clouds organized as 2d images may be produced by
# camera depth sensors such as stereo or time-of-flight.

# Time of sensor data acquisition, and the coordinate frame ID (for 3d
# points).
Header header

# 2D structure of the point cloud. If the cloud is unordered, height is
# 1 and width is the length of the point cloud.
uint32 height
uint32 width

# Describes the channels and their layout in the binary data blob.
PointField[] fields

bool    is_bigendian # Is this data bigendian?
uint32  point_step   # Length of a point in bytes
uint32  row_step     # Length of a row in bytes
uint8[] data         # Actual point data, size is (row_step*height)

bool is_dense        # True if there are no invalid points

================================================================================
MSG: sensor_msgs/PointField
# This message holds the description of one point entry in the
# PointCloud2 message format.
uint8 INT8    = 1
uint8 UINT8   = 2
uint8 INT16   = 3
uint8 UINT16  = 4
uint8 INT32   = 5
uint8 UINT32  = 6
uint8 FLOAT32 = 7
uint8 FLOAT64 = 8

string name      # Name of field
uint32 offset    # Offset from start of point struct
uint8  datatype  # Datatype enumeration, see above
uint32 count     # How many elements in the field

================================================================================
MSG: omnimapper_msgs/Centroid
Header header
string platform
geometry_msgs/Point centroid

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
"""
  __slots__ = ['header','platform','frame','symbol','cloud','centroid']
  _slot_types = ['std_msgs/Header','string','omnimapper_msgs/FrameNotification','omnimapper_msgs/GtsamSymbol','omnimapper_msgs/StampedPointCloud','omnimapper_msgs/Centroid']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       header,platform,frame,symbol,cloud,centroid

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(FrameRecord, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.platform is None:
        self.platform = ''
      if self.frame is None:
        self.frame = omnimapper_msgs.msg.FrameNotification()
      if self.symbol is None:
        self.symbol = omnimapper_msgs.msg.GtsamSymbol()
      if self.cloud is None:
        self.cloud = omnimapper_msgs.msg.StampedPointCloud()
      if self.centroid is None:
        self.centroid = omnimapper_msgs.msg.Centroid()
    else:
      self.header = std_msgs.msg.Header()
      self.platform = ''
      self.frame = omnimapper_msgs.msg.FrameNotification()
      self.symbol = omnimapper_msgs.msg.GtsamSymbol()
      self.cloud = omnimapper_msgs.msg.StampedPointCloud()
      self.centroid = omnimapper_msgs.msg.Centroid()

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
      _x = self.platform
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_3I().pack(_x.frame.header.seq, _x.frame.header.stamp.secs, _x.frame.header.stamp.nsecs))
      _x = self.frame.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.frame.platform
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_2i3I().pack(_x.frame.type, _x.frame.mode, _x.frame.pose.header.seq, _x.frame.pose.header.stamp.secs, _x.frame.pose.header.stamp.nsecs))
      _x = self.frame.pose.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.frame.pose.child_frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_7dBQ3I().pack(_x.frame.pose.transform.translation.x, _x.frame.pose.transform.translation.y, _x.frame.pose.transform.translation.z, _x.frame.pose.transform.rotation.x, _x.frame.pose.transform.rotation.y, _x.frame.pose.transform.rotation.z, _x.frame.pose.transform.rotation.w, _x.symbol.symbol, _x.symbol.index, _x.cloud.transform.header.seq, _x.cloud.transform.header.stamp.secs, _x.cloud.transform.header.stamp.nsecs))
      _x = self.cloud.transform.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.cloud.transform.child_frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_7d().pack(_x.cloud.transform.transform.translation.x, _x.cloud.transform.transform.translation.y, _x.cloud.transform.transform.translation.z, _x.cloud.transform.transform.rotation.x, _x.cloud.transform.transform.rotation.y, _x.cloud.transform.transform.rotation.z, _x.cloud.transform.transform.rotation.w))
      _x = self.cloud.platform
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_3I().pack(_x.cloud.cloud.header.seq, _x.cloud.cloud.header.stamp.secs, _x.cloud.cloud.header.stamp.nsecs))
      _x = self.cloud.cloud.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_2I().pack(_x.cloud.cloud.height, _x.cloud.cloud.width))
      length = len(self.cloud.cloud.fields)
      buff.write(_struct_I.pack(length))
      for val1 in self.cloud.cloud.fields:
        _x = val1.name
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = val1
        buff.write(_get_struct_IBI().pack(_x.offset, _x.datatype, _x.count))
      _x = self
      buff.write(_get_struct_B2I().pack(_x.cloud.cloud.is_bigendian, _x.cloud.cloud.point_step, _x.cloud.cloud.row_step))
      _x = self.cloud.cloud.data
      length = len(_x)
      # - if encoded as a list instead, serialize as bytes instead of string
      if type(_x) in [list, tuple]:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_B3I().pack(_x.cloud.cloud.is_dense, _x.centroid.header.seq, _x.centroid.header.stamp.secs, _x.centroid.header.stamp.nsecs))
      _x = self.centroid.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.centroid.platform
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_3d().pack(_x.centroid.centroid.x, _x.centroid.centroid.y, _x.centroid.centroid.z))
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
      if self.frame is None:
        self.frame = omnimapper_msgs.msg.FrameNotification()
      if self.symbol is None:
        self.symbol = omnimapper_msgs.msg.GtsamSymbol()
      if self.cloud is None:
        self.cloud = omnimapper_msgs.msg.StampedPointCloud()
      if self.centroid is None:
        self.centroid = omnimapper_msgs.msg.Centroid()
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
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.platform = str[start:end].decode('utf-8')
      else:
        self.platform = str[start:end]
      _x = self
      start = end
      end += 12
      (_x.frame.header.seq, _x.frame.header.stamp.secs, _x.frame.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.frame.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.frame.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.frame.platform = str[start:end].decode('utf-8')
      else:
        self.frame.platform = str[start:end]
      _x = self
      start = end
      end += 20
      (_x.frame.type, _x.frame.mode, _x.frame.pose.header.seq, _x.frame.pose.header.stamp.secs, _x.frame.pose.header.stamp.nsecs,) = _get_struct_2i3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.frame.pose.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.frame.pose.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.frame.pose.child_frame_id = str[start:end].decode('utf-8')
      else:
        self.frame.pose.child_frame_id = str[start:end]
      _x = self
      start = end
      end += 77
      (_x.frame.pose.transform.translation.x, _x.frame.pose.transform.translation.y, _x.frame.pose.transform.translation.z, _x.frame.pose.transform.rotation.x, _x.frame.pose.transform.rotation.y, _x.frame.pose.transform.rotation.z, _x.frame.pose.transform.rotation.w, _x.symbol.symbol, _x.symbol.index, _x.cloud.transform.header.seq, _x.cloud.transform.header.stamp.secs, _x.cloud.transform.header.stamp.nsecs,) = _get_struct_7dBQ3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.cloud.transform.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.cloud.transform.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.cloud.transform.child_frame_id = str[start:end].decode('utf-8')
      else:
        self.cloud.transform.child_frame_id = str[start:end]
      _x = self
      start = end
      end += 56
      (_x.cloud.transform.transform.translation.x, _x.cloud.transform.transform.translation.y, _x.cloud.transform.transform.translation.z, _x.cloud.transform.transform.rotation.x, _x.cloud.transform.transform.rotation.y, _x.cloud.transform.transform.rotation.z, _x.cloud.transform.transform.rotation.w,) = _get_struct_7d().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.cloud.platform = str[start:end].decode('utf-8')
      else:
        self.cloud.platform = str[start:end]
      _x = self
      start = end
      end += 12
      (_x.cloud.cloud.header.seq, _x.cloud.cloud.header.stamp.secs, _x.cloud.cloud.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.cloud.cloud.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.cloud.cloud.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 8
      (_x.cloud.cloud.height, _x.cloud.cloud.width,) = _get_struct_2I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.cloud.cloud.fields = []
      for i in range(0, length):
        val1 = sensor_msgs.msg.PointField()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.name = str[start:end].decode('utf-8')
        else:
          val1.name = str[start:end]
        _x = val1
        start = end
        end += 9
        (_x.offset, _x.datatype, _x.count,) = _get_struct_IBI().unpack(str[start:end])
        self.cloud.cloud.fields.append(val1)
      _x = self
      start = end
      end += 9
      (_x.cloud.cloud.is_bigendian, _x.cloud.cloud.point_step, _x.cloud.cloud.row_step,) = _get_struct_B2I().unpack(str[start:end])
      self.cloud.cloud.is_bigendian = bool(self.cloud.cloud.is_bigendian)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.cloud.cloud.data = str[start:end]
      _x = self
      start = end
      end += 13
      (_x.cloud.cloud.is_dense, _x.centroid.header.seq, _x.centroid.header.stamp.secs, _x.centroid.header.stamp.nsecs,) = _get_struct_B3I().unpack(str[start:end])
      self.cloud.cloud.is_dense = bool(self.cloud.cloud.is_dense)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.centroid.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.centroid.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.centroid.platform = str[start:end].decode('utf-8')
      else:
        self.centroid.platform = str[start:end]
      _x = self
      start = end
      end += 24
      (_x.centroid.centroid.x, _x.centroid.centroid.y, _x.centroid.centroid.z,) = _get_struct_3d().unpack(str[start:end])
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
      _x = self.platform
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_3I().pack(_x.frame.header.seq, _x.frame.header.stamp.secs, _x.frame.header.stamp.nsecs))
      _x = self.frame.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.frame.platform
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_2i3I().pack(_x.frame.type, _x.frame.mode, _x.frame.pose.header.seq, _x.frame.pose.header.stamp.secs, _x.frame.pose.header.stamp.nsecs))
      _x = self.frame.pose.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.frame.pose.child_frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_7dBQ3I().pack(_x.frame.pose.transform.translation.x, _x.frame.pose.transform.translation.y, _x.frame.pose.transform.translation.z, _x.frame.pose.transform.rotation.x, _x.frame.pose.transform.rotation.y, _x.frame.pose.transform.rotation.z, _x.frame.pose.transform.rotation.w, _x.symbol.symbol, _x.symbol.index, _x.cloud.transform.header.seq, _x.cloud.transform.header.stamp.secs, _x.cloud.transform.header.stamp.nsecs))
      _x = self.cloud.transform.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.cloud.transform.child_frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_7d().pack(_x.cloud.transform.transform.translation.x, _x.cloud.transform.transform.translation.y, _x.cloud.transform.transform.translation.z, _x.cloud.transform.transform.rotation.x, _x.cloud.transform.transform.rotation.y, _x.cloud.transform.transform.rotation.z, _x.cloud.transform.transform.rotation.w))
      _x = self.cloud.platform
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_3I().pack(_x.cloud.cloud.header.seq, _x.cloud.cloud.header.stamp.secs, _x.cloud.cloud.header.stamp.nsecs))
      _x = self.cloud.cloud.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_2I().pack(_x.cloud.cloud.height, _x.cloud.cloud.width))
      length = len(self.cloud.cloud.fields)
      buff.write(_struct_I.pack(length))
      for val1 in self.cloud.cloud.fields:
        _x = val1.name
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = val1
        buff.write(_get_struct_IBI().pack(_x.offset, _x.datatype, _x.count))
      _x = self
      buff.write(_get_struct_B2I().pack(_x.cloud.cloud.is_bigendian, _x.cloud.cloud.point_step, _x.cloud.cloud.row_step))
      _x = self.cloud.cloud.data
      length = len(_x)
      # - if encoded as a list instead, serialize as bytes instead of string
      if type(_x) in [list, tuple]:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_B3I().pack(_x.cloud.cloud.is_dense, _x.centroid.header.seq, _x.centroid.header.stamp.secs, _x.centroid.header.stamp.nsecs))
      _x = self.centroid.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.centroid.platform
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_3d().pack(_x.centroid.centroid.x, _x.centroid.centroid.y, _x.centroid.centroid.z))
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
      if self.frame is None:
        self.frame = omnimapper_msgs.msg.FrameNotification()
      if self.symbol is None:
        self.symbol = omnimapper_msgs.msg.GtsamSymbol()
      if self.cloud is None:
        self.cloud = omnimapper_msgs.msg.StampedPointCloud()
      if self.centroid is None:
        self.centroid = omnimapper_msgs.msg.Centroid()
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
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.platform = str[start:end].decode('utf-8')
      else:
        self.platform = str[start:end]
      _x = self
      start = end
      end += 12
      (_x.frame.header.seq, _x.frame.header.stamp.secs, _x.frame.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.frame.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.frame.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.frame.platform = str[start:end].decode('utf-8')
      else:
        self.frame.platform = str[start:end]
      _x = self
      start = end
      end += 20
      (_x.frame.type, _x.frame.mode, _x.frame.pose.header.seq, _x.frame.pose.header.stamp.secs, _x.frame.pose.header.stamp.nsecs,) = _get_struct_2i3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.frame.pose.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.frame.pose.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.frame.pose.child_frame_id = str[start:end].decode('utf-8')
      else:
        self.frame.pose.child_frame_id = str[start:end]
      _x = self
      start = end
      end += 77
      (_x.frame.pose.transform.translation.x, _x.frame.pose.transform.translation.y, _x.frame.pose.transform.translation.z, _x.frame.pose.transform.rotation.x, _x.frame.pose.transform.rotation.y, _x.frame.pose.transform.rotation.z, _x.frame.pose.transform.rotation.w, _x.symbol.symbol, _x.symbol.index, _x.cloud.transform.header.seq, _x.cloud.transform.header.stamp.secs, _x.cloud.transform.header.stamp.nsecs,) = _get_struct_7dBQ3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.cloud.transform.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.cloud.transform.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.cloud.transform.child_frame_id = str[start:end].decode('utf-8')
      else:
        self.cloud.transform.child_frame_id = str[start:end]
      _x = self
      start = end
      end += 56
      (_x.cloud.transform.transform.translation.x, _x.cloud.transform.transform.translation.y, _x.cloud.transform.transform.translation.z, _x.cloud.transform.transform.rotation.x, _x.cloud.transform.transform.rotation.y, _x.cloud.transform.transform.rotation.z, _x.cloud.transform.transform.rotation.w,) = _get_struct_7d().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.cloud.platform = str[start:end].decode('utf-8')
      else:
        self.cloud.platform = str[start:end]
      _x = self
      start = end
      end += 12
      (_x.cloud.cloud.header.seq, _x.cloud.cloud.header.stamp.secs, _x.cloud.cloud.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.cloud.cloud.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.cloud.cloud.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 8
      (_x.cloud.cloud.height, _x.cloud.cloud.width,) = _get_struct_2I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.cloud.cloud.fields = []
      for i in range(0, length):
        val1 = sensor_msgs.msg.PointField()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.name = str[start:end].decode('utf-8')
        else:
          val1.name = str[start:end]
        _x = val1
        start = end
        end += 9
        (_x.offset, _x.datatype, _x.count,) = _get_struct_IBI().unpack(str[start:end])
        self.cloud.cloud.fields.append(val1)
      _x = self
      start = end
      end += 9
      (_x.cloud.cloud.is_bigendian, _x.cloud.cloud.point_step, _x.cloud.cloud.row_step,) = _get_struct_B2I().unpack(str[start:end])
      self.cloud.cloud.is_bigendian = bool(self.cloud.cloud.is_bigendian)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.cloud.cloud.data = str[start:end]
      _x = self
      start = end
      end += 13
      (_x.cloud.cloud.is_dense, _x.centroid.header.seq, _x.centroid.header.stamp.secs, _x.centroid.header.stamp.nsecs,) = _get_struct_B3I().unpack(str[start:end])
      self.cloud.cloud.is_dense = bool(self.cloud.cloud.is_dense)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.centroid.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.centroid.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.centroid.platform = str[start:end].decode('utf-8')
      else:
        self.centroid.platform = str[start:end]
      _x = self
      start = end
      end += 24
      (_x.centroid.centroid.x, _x.centroid.centroid.y, _x.centroid.centroid.z,) = _get_struct_3d().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_7dBQ3I = None
def _get_struct_7dBQ3I():
    global _struct_7dBQ3I
    if _struct_7dBQ3I is None:
        _struct_7dBQ3I = struct.Struct("<7dBQ3I")
    return _struct_7dBQ3I
_struct_IBI = None
def _get_struct_IBI():
    global _struct_IBI
    if _struct_IBI is None:
        _struct_IBI = struct.Struct("<IBI")
    return _struct_IBI
_struct_7d = None
def _get_struct_7d():
    global _struct_7d
    if _struct_7d is None:
        _struct_7d = struct.Struct("<7d")
    return _struct_7d
_struct_2i3I = None
def _get_struct_2i3I():
    global _struct_2i3I
    if _struct_2i3I is None:
        _struct_2i3I = struct.Struct("<2i3I")
    return _struct_2i3I
_struct_3I = None
def _get_struct_3I():
    global _struct_3I
    if _struct_3I is None:
        _struct_3I = struct.Struct("<3I")
    return _struct_3I
_struct_B3I = None
def _get_struct_B3I():
    global _struct_B3I
    if _struct_B3I is None:
        _struct_B3I = struct.Struct("<B3I")
    return _struct_B3I
_struct_B2I = None
def _get_struct_B2I():
    global _struct_B2I
    if _struct_B2I is None:
        _struct_B2I = struct.Struct("<B2I")
    return _struct_B2I
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
