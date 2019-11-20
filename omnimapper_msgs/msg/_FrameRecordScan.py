# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from omnimapper_msgs/FrameRecordScan.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import omnimapper_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg
import std_msgs.msg

class FrameRecordScan(genpy.Message):
  _md5sum = "4898872f9f9a4d9eb25315515cd09171"
  _type = "omnimapper_msgs/FrameRecordScan"
  _has_header = True #flag to mark the presence of a Header object
  _full_text = """Header header
string platform
omnimapper_msgs/FrameNotification frame
omnimapper_msgs/GtsamSymbol symbol
omnimapper_msgs/StampedLaserScan scan
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
MSG: omnimapper_msgs/StampedLaserScan
# The pose and time stamp of point cloud
geometry_msgs/TransformStamped transformStart
geometry_msgs/TransformStamped transformEnd
# The platform source of this point cloud
string platform
# The associated laser scan
sensor_msgs/LaserScan scan


================================================================================
MSG: sensor_msgs/LaserScan
# Single scan from a planar laser range-finder
#
# If you have another ranging device with different behavior (e.g. a sonar
# array), please find or create a different message, since applications
# will make fairly laser-specific assumptions about this data

Header header            # timestamp in the header is the acquisition time of 
                         # the first ray in the scan.
                         #
                         # in frame frame_id, angles are measured around 
                         # the positive Z axis (counterclockwise, if Z is up)
                         # with zero angle being forward along the x axis
                         
float32 angle_min        # start angle of the scan [rad]
float32 angle_max        # end angle of the scan [rad]
float32 angle_increment  # angular distance between measurements [rad]

float32 time_increment   # time between measurements [seconds] - if your scanner
                         # is moving, this will be used in interpolating position
                         # of 3d points
float32 scan_time        # time between scans [seconds]

float32 range_min        # minimum range value [m]
float32 range_max        # maximum range value [m]

float32[] ranges         # range data [m] (Note: values < range_min or > range_max should be discarded)
float32[] intensities    # intensity data [device-specific units].  If your
                         # device does not provide intensities, please leave
                         # the array empty.

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
  __slots__ = ['header','platform','frame','symbol','scan','centroid']
  _slot_types = ['std_msgs/Header','string','omnimapper_msgs/FrameNotification','omnimapper_msgs/GtsamSymbol','omnimapper_msgs/StampedLaserScan','omnimapper_msgs/Centroid']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       header,platform,frame,symbol,scan,centroid

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(FrameRecordScan, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.platform is None:
        self.platform = ''
      if self.frame is None:
        self.frame = omnimapper_msgs.msg.FrameNotification()
      if self.symbol is None:
        self.symbol = omnimapper_msgs.msg.GtsamSymbol()
      if self.scan is None:
        self.scan = omnimapper_msgs.msg.StampedLaserScan()
      if self.centroid is None:
        self.centroid = omnimapper_msgs.msg.Centroid()
    else:
      self.header = std_msgs.msg.Header()
      self.platform = ''
      self.frame = omnimapper_msgs.msg.FrameNotification()
      self.symbol = omnimapper_msgs.msg.GtsamSymbol()
      self.scan = omnimapper_msgs.msg.StampedLaserScan()
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
      buff.write(_get_struct_7dBQ3I().pack(_x.frame.pose.transform.translation.x, _x.frame.pose.transform.translation.y, _x.frame.pose.transform.translation.z, _x.frame.pose.transform.rotation.x, _x.frame.pose.transform.rotation.y, _x.frame.pose.transform.rotation.z, _x.frame.pose.transform.rotation.w, _x.symbol.symbol, _x.symbol.index, _x.scan.transformStart.header.seq, _x.scan.transformStart.header.stamp.secs, _x.scan.transformStart.header.stamp.nsecs))
      _x = self.scan.transformStart.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.scan.transformStart.child_frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_7d3I().pack(_x.scan.transformStart.transform.translation.x, _x.scan.transformStart.transform.translation.y, _x.scan.transformStart.transform.translation.z, _x.scan.transformStart.transform.rotation.x, _x.scan.transformStart.transform.rotation.y, _x.scan.transformStart.transform.rotation.z, _x.scan.transformStart.transform.rotation.w, _x.scan.transformEnd.header.seq, _x.scan.transformEnd.header.stamp.secs, _x.scan.transformEnd.header.stamp.nsecs))
      _x = self.scan.transformEnd.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.scan.transformEnd.child_frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_7d().pack(_x.scan.transformEnd.transform.translation.x, _x.scan.transformEnd.transform.translation.y, _x.scan.transformEnd.transform.translation.z, _x.scan.transformEnd.transform.rotation.x, _x.scan.transformEnd.transform.rotation.y, _x.scan.transformEnd.transform.rotation.z, _x.scan.transformEnd.transform.rotation.w))
      _x = self.scan.platform
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_3I().pack(_x.scan.scan.header.seq, _x.scan.scan.header.stamp.secs, _x.scan.scan.header.stamp.nsecs))
      _x = self.scan.scan.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_7f().pack(_x.scan.scan.angle_min, _x.scan.scan.angle_max, _x.scan.scan.angle_increment, _x.scan.scan.time_increment, _x.scan.scan.scan_time, _x.scan.scan.range_min, _x.scan.scan.range_max))
      length = len(self.scan.scan.ranges)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.pack(pattern, *self.scan.scan.ranges))
      length = len(self.scan.scan.intensities)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.pack(pattern, *self.scan.scan.intensities))
      _x = self
      buff.write(_get_struct_3I().pack(_x.centroid.header.seq, _x.centroid.header.stamp.secs, _x.centroid.header.stamp.nsecs))
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
      if self.scan is None:
        self.scan = omnimapper_msgs.msg.StampedLaserScan()
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
      (_x.frame.pose.transform.translation.x, _x.frame.pose.transform.translation.y, _x.frame.pose.transform.translation.z, _x.frame.pose.transform.rotation.x, _x.frame.pose.transform.rotation.y, _x.frame.pose.transform.rotation.z, _x.frame.pose.transform.rotation.w, _x.symbol.symbol, _x.symbol.index, _x.scan.transformStart.header.seq, _x.scan.transformStart.header.stamp.secs, _x.scan.transformStart.header.stamp.nsecs,) = _get_struct_7dBQ3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.scan.transformStart.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.scan.transformStart.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.scan.transformStart.child_frame_id = str[start:end].decode('utf-8')
      else:
        self.scan.transformStart.child_frame_id = str[start:end]
      _x = self
      start = end
      end += 68
      (_x.scan.transformStart.transform.translation.x, _x.scan.transformStart.transform.translation.y, _x.scan.transformStart.transform.translation.z, _x.scan.transformStart.transform.rotation.x, _x.scan.transformStart.transform.rotation.y, _x.scan.transformStart.transform.rotation.z, _x.scan.transformStart.transform.rotation.w, _x.scan.transformEnd.header.seq, _x.scan.transformEnd.header.stamp.secs, _x.scan.transformEnd.header.stamp.nsecs,) = _get_struct_7d3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.scan.transformEnd.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.scan.transformEnd.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.scan.transformEnd.child_frame_id = str[start:end].decode('utf-8')
      else:
        self.scan.transformEnd.child_frame_id = str[start:end]
      _x = self
      start = end
      end += 56
      (_x.scan.transformEnd.transform.translation.x, _x.scan.transformEnd.transform.translation.y, _x.scan.transformEnd.transform.translation.z, _x.scan.transformEnd.transform.rotation.x, _x.scan.transformEnd.transform.rotation.y, _x.scan.transformEnd.transform.rotation.z, _x.scan.transformEnd.transform.rotation.w,) = _get_struct_7d().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.scan.platform = str[start:end].decode('utf-8')
      else:
        self.scan.platform = str[start:end]
      _x = self
      start = end
      end += 12
      (_x.scan.scan.header.seq, _x.scan.scan.header.stamp.secs, _x.scan.scan.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.scan.scan.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.scan.scan.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 28
      (_x.scan.scan.angle_min, _x.scan.scan.angle_max, _x.scan.scan.angle_increment, _x.scan.scan.time_increment, _x.scan.scan.scan_time, _x.scan.scan.range_min, _x.scan.scan.range_max,) = _get_struct_7f().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.scan.scan.ranges = struct.unpack(pattern, str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.scan.scan.intensities = struct.unpack(pattern, str[start:end])
      _x = self
      start = end
      end += 12
      (_x.centroid.header.seq, _x.centroid.header.stamp.secs, _x.centroid.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
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
      buff.write(_get_struct_7dBQ3I().pack(_x.frame.pose.transform.translation.x, _x.frame.pose.transform.translation.y, _x.frame.pose.transform.translation.z, _x.frame.pose.transform.rotation.x, _x.frame.pose.transform.rotation.y, _x.frame.pose.transform.rotation.z, _x.frame.pose.transform.rotation.w, _x.symbol.symbol, _x.symbol.index, _x.scan.transformStart.header.seq, _x.scan.transformStart.header.stamp.secs, _x.scan.transformStart.header.stamp.nsecs))
      _x = self.scan.transformStart.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.scan.transformStart.child_frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_7d3I().pack(_x.scan.transformStart.transform.translation.x, _x.scan.transformStart.transform.translation.y, _x.scan.transformStart.transform.translation.z, _x.scan.transformStart.transform.rotation.x, _x.scan.transformStart.transform.rotation.y, _x.scan.transformStart.transform.rotation.z, _x.scan.transformStart.transform.rotation.w, _x.scan.transformEnd.header.seq, _x.scan.transformEnd.header.stamp.secs, _x.scan.transformEnd.header.stamp.nsecs))
      _x = self.scan.transformEnd.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.scan.transformEnd.child_frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_7d().pack(_x.scan.transformEnd.transform.translation.x, _x.scan.transformEnd.transform.translation.y, _x.scan.transformEnd.transform.translation.z, _x.scan.transformEnd.transform.rotation.x, _x.scan.transformEnd.transform.rotation.y, _x.scan.transformEnd.transform.rotation.z, _x.scan.transformEnd.transform.rotation.w))
      _x = self.scan.platform
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_3I().pack(_x.scan.scan.header.seq, _x.scan.scan.header.stamp.secs, _x.scan.scan.header.stamp.nsecs))
      _x = self.scan.scan.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_7f().pack(_x.scan.scan.angle_min, _x.scan.scan.angle_max, _x.scan.scan.angle_increment, _x.scan.scan.time_increment, _x.scan.scan.scan_time, _x.scan.scan.range_min, _x.scan.scan.range_max))
      length = len(self.scan.scan.ranges)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.scan.scan.ranges.tostring())
      length = len(self.scan.scan.intensities)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.scan.scan.intensities.tostring())
      _x = self
      buff.write(_get_struct_3I().pack(_x.centroid.header.seq, _x.centroid.header.stamp.secs, _x.centroid.header.stamp.nsecs))
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
      if self.scan is None:
        self.scan = omnimapper_msgs.msg.StampedLaserScan()
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
      (_x.frame.pose.transform.translation.x, _x.frame.pose.transform.translation.y, _x.frame.pose.transform.translation.z, _x.frame.pose.transform.rotation.x, _x.frame.pose.transform.rotation.y, _x.frame.pose.transform.rotation.z, _x.frame.pose.transform.rotation.w, _x.symbol.symbol, _x.symbol.index, _x.scan.transformStart.header.seq, _x.scan.transformStart.header.stamp.secs, _x.scan.transformStart.header.stamp.nsecs,) = _get_struct_7dBQ3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.scan.transformStart.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.scan.transformStart.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.scan.transformStart.child_frame_id = str[start:end].decode('utf-8')
      else:
        self.scan.transformStart.child_frame_id = str[start:end]
      _x = self
      start = end
      end += 68
      (_x.scan.transformStart.transform.translation.x, _x.scan.transformStart.transform.translation.y, _x.scan.transformStart.transform.translation.z, _x.scan.transformStart.transform.rotation.x, _x.scan.transformStart.transform.rotation.y, _x.scan.transformStart.transform.rotation.z, _x.scan.transformStart.transform.rotation.w, _x.scan.transformEnd.header.seq, _x.scan.transformEnd.header.stamp.secs, _x.scan.transformEnd.header.stamp.nsecs,) = _get_struct_7d3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.scan.transformEnd.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.scan.transformEnd.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.scan.transformEnd.child_frame_id = str[start:end].decode('utf-8')
      else:
        self.scan.transformEnd.child_frame_id = str[start:end]
      _x = self
      start = end
      end += 56
      (_x.scan.transformEnd.transform.translation.x, _x.scan.transformEnd.transform.translation.y, _x.scan.transformEnd.transform.translation.z, _x.scan.transformEnd.transform.rotation.x, _x.scan.transformEnd.transform.rotation.y, _x.scan.transformEnd.transform.rotation.z, _x.scan.transformEnd.transform.rotation.w,) = _get_struct_7d().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.scan.platform = str[start:end].decode('utf-8')
      else:
        self.scan.platform = str[start:end]
      _x = self
      start = end
      end += 12
      (_x.scan.scan.header.seq, _x.scan.scan.header.stamp.secs, _x.scan.scan.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.scan.scan.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.scan.scan.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 28
      (_x.scan.scan.angle_min, _x.scan.scan.angle_max, _x.scan.scan.angle_increment, _x.scan.scan.time_increment, _x.scan.scan.scan_time, _x.scan.scan.range_min, _x.scan.scan.range_max,) = _get_struct_7f().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.scan.scan.ranges = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.scan.scan.intensities = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
      _x = self
      start = end
      end += 12
      (_x.centroid.header.seq, _x.centroid.header.stamp.secs, _x.centroid.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
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
_struct_7f = None
def _get_struct_7f():
    global _struct_7f
    if _struct_7f is None:
        _struct_7f = struct.Struct("<7f")
    return _struct_7f
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
_struct_7d3I = None
def _get_struct_7d3I():
    global _struct_7d3I
    if _struct_7d3I is None:
        _struct_7d3I = struct.Struct("<7d3I")
    return _struct_7d3I
_struct_3d = None
def _get_struct_3d():
    global _struct_3d
    if _struct_3d is None:
        _struct_3d = struct.Struct("<3d")
    return _struct_3d
