"""Route isolated SSL backbones to their matching runtime and feature exporter."""


def rewrite_command(command, arm):
    routed = list(command)
    if len(routed) < 2:
        return routed
    if routed[1] == arm.get("script"):
        routed[0] = arm.get("python", routed[0])
    elif routed[1] == "scripts/export_overlay_features.py" and arm.get("exporter"):
        routed[0] = arm.get("python", routed[0])
        routed[1] = arm["exporter"]
    return routed
