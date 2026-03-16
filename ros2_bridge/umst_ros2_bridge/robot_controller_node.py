"""
UMST Robot Controller Node (UR10e)
==================================
Demonstration node showing how a UR10e robotic fabrication controller integrates
with the UMST thermodynamic gate via ROS2. Subscribes to gate decisions
and translates ACK'd proposals into robot commands.

Safety invariant: NO command is sent to hardware unless the gate returns
admissible=true. This is the ROS2-side enforcement of Constitutional Physics.

Architecture:
    /umst/decision (from gate_bridge) → RobotControllerNode
        → if admissible: /robot/cmd (geometry_msgs/Twist or custom)
        → if rejected:   log + /umst/replan_request

Usage:
    ros2 run umst_ros2_bridge robot_controller
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


class RobotControllerNode(Node):
    """
    Receives gate decisions and issues robot commands only for admissible transitions.
    """

    def __init__(self):
        super().__init__("umst_robot_controller")

        self.declare_parameter("dry_run", True)

        self.dry_run = self.get_parameter("dry_run").get_parameter_value().bool_value

        self.sub_decision = self.create_subscription(
            String, "/umst/decision", self.on_decision, 10
        )

        self.pub_cmd = self.create_publisher(String, "/robot/cmd", 10)
        self.pub_replan = self.create_publisher(String, "/umst/replan_request", 10)

        self.commands_issued = 0
        self.rejections_handled = 0

        mode = "DRY RUN" if self.dry_run else "LIVE"
        self.get_logger().info(f"UR10e robot controller started [{mode}]")

    def on_decision(self, msg: String):
        try:
            decision = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Malformed gate decision JSON")
            return

        if decision.get("admissible", False):
            self.commands_issued += 1
            command_id = f"{decision.get('plan_id', 'plan')}-{self.commands_issued}"
            cmd = String()
            cmd.data = json.dumps({
                "command_id": command_id,
                "plan_id": decision.get("plan_id", "unknown_plan"),
                "action": "execute",
                "source": decision.get("source", "system"),
                "required_gate_decision_id": decision.get("decision_id", "unknown_decision"),
                "physics_strength": decision.get("physics_strength", 0.0),
                "hydration_degree": decision.get("hydration_degree", 0.0),
                "sequence": self.commands_issued,
                "safety_envelope": {  # UR10e defaults
                    "workspace_radius_m": decision.get("workspace_radius_m", 1.3),
                    "max_tcp_speed_mps": decision.get("max_tcp_speed_mps", 0.25),
                    "max_payload_kg": decision.get("max_payload_kg", 10.0),
                    "min_confidence": decision.get("min_confidence", 0.3),
                },
            })
            if not self.dry_run:
                self.pub_cmd.publish(cmd)
            self.get_logger().info(
                f"{'[DRY] ' if self.dry_run else ''}CMD #{self.commands_issued}: "
                f"fc={decision.get('physics_strength', 0):.1f} MPa"
            )
        else:
            self.rejections_handled += 1
            replan = String()
            replan.data = json.dumps({
                "action": "replan",
                "violation": decision.get("violation", "unknown"),
                "verdict": decision.get("verdict", ""),
            })
            self.pub_replan.publish(replan)
            self.get_logger().warning(
                f"NACK → replan requested: {decision.get('violation', 'unknown')}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"Controller: {node.commands_issued} commands, "
            f"{node.rejections_handled} replans"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
