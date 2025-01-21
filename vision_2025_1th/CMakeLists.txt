cmake_minimum_required(VERSION 3.8)
project(vision_2025_1th)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME} "msg/Vision.msg")
ament_export_dependencies(rosidl_default_runtime)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

install(DIRECTORY msg DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
    scripts/1th.py
    scripts/cv_tools.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()