import xml.etree.ElementTree as ET
from functools import partial
from gz.transport13 import Node
import gz.msgs10.contacts_pb2 as contacts_pb2
import time
import gz.msgs10.laserscan_pb2 as laserscan_pb2
import numpy as np
import os
class SensorReading():
    def __init__(self, instance_id):
        
        self.target_names = []
        self.world_name = None
        self.crashed = False
        self.lidar_data = np.zeros(6, dtype=np.float32)
        self.started = False
        self.instance_id = instance_id
        # Store nodes created in __init__
        self.lidar_nodes = [] # List to hold lidar nodes
        self.lidar_subscriptions = []
        self.contact_nodes = []
        self.contact_subscriptions = []

    def start_lidars(self):

        self.front_node = Node()
        self.back_node  = Node()
        self.left_node  = Node()
        self.right_node = Node()
        self.down_node  = Node()
        self.up_node    = Node()

        self.lidar_nodes.append(self.down_node)
        self.lidar_nodes.append(self.up_node)
        self.lidar_nodes.append(self.front_node)
        self.lidar_nodes.append(self.back_node)
        self.lidar_nodes.append(self.left_node)
        self.lidar_nodes.append(self.right_node)


        self.lidar_subscriptions.append(self.down_node.subscribe(topic=f"/__model__/down", msg_type=laserscan_pb2.LaserScan, callback=self.make_lidar_callback("down", 16)))
        self.lidar_subscriptions.append(self.up_node.subscribe(topic=f"/__model__/up"  , msg_type=laserscan_pb2.LaserScan, callback=self.make_lidar_callback("up", 20)))
        self.lidar_subscriptions.append(self.front_node.subscribe(topic=f"/__model__/front", msg_type=laserscan_pb2.LaserScan, callback=self.make_lidar_callback("front", 0)))
        self.lidar_subscriptions.append(self.back_node.subscribe (topic=f"/__model__/back", msg_type=laserscan_pb2.LaserScan, callback=self.make_lidar_callback("back", 4)))
        self.lidar_subscriptions.append(self.left_node.subscribe (topic=f"/__model__/left", msg_type=laserscan_pb2.LaserScan, callback=self.make_lidar_callback("left",8)))
        self.lidar_subscriptions.append(self.right_node.subscribe(topic=f"/__model__/right", msg_type=laserscan_pb2.LaserScan, callback=self.make_lidar_callback("right",12)))

    def make_lidar_callback(self, direction, index):
        def _callback(msg):
            if msg.ranges:
                output = np.clip(np.nan_to_num(msg.ranges[np.argmin(msg.ranges)],posinf=4.0, neginf=0.0), 0.0, 4.0)
                self.lidar_data[(int(index / 4))] = output

                
                """
                if num_ranges >= 4:
                    # np.clip(value, min, max) ensures values are within a range
                    # Lidar values usually go up to a max range, e.g., 5m.
                    # Setting posinf to 5 and neginf to 0 is common for normalization.
                    self.lidar_data[index:index+4] = np.clip(
                        np.nan_to_num(list(msg.ranges)[:4], posinf=5.0, neginf=0.0), 0.0, 5.0
                    ) 
                    #if direction ==  "right":
                        #print("Down: ", self.lidar_data[16:20])
                        #print("Up: ", self.lidar_data[20:24])
                        #print("Front: ", self.lidar_data[0:4])
                        #print("Back: ", self.lidar_data[4:8])
                        #print("Left: ", self.lidar_data[8:12])
                        #print("Right: ", self.lidar_data[12:16])
                """

        return _callback

    def _parse_sdf(self, sdf_path):
        # Clear previous target names and nodes if parsing a new SDF
        self.target_names = []
        self.world_name = None

        # IMPORTANT: Clean up existing contact sensor nodes/subscriptions before creating new ones
        self.stop_touch_listeners() # Call cleanup here!

        tree = ET.parse(sdf_path)
        root = tree.getroot()

        keywords = ["wall", "ceiling", "obstacle", "floor"] # Added "floor" for contact sensor on floor

        world_elem = root.find("world")
        if world_elem is not None and "name" in world_elem.attrib:
            self.world_name = world_elem.attrib["name"]

        for model in root.findall(".//model"):
            model_name = model.attrib.get("name", "").lower()

            for link in model.findall("link"):
                link_name = link.attrib.get("name", "").lower()
                if any(keyword in link_name for keyword in keywords):
                    self.target_names.append((link_name, model_name))

            if any(keyword in model_name for keyword in keywords):
                self.target_names.append((model_name, model_name))

    def start_touch_listeners(self):
        # This method should only create new nodes/subscriptions
        # Cleanup of old ones should happen in stop_touch_listener
        
        print(self.target_names)
        for link_name, model_name in self.target_names:
            topic = f"/world/{self.world_name}/model/{model_name}/link/{link_name}/sensor/sensor_contact/contact"
            
            node = Node()
            sub = node.subscribe(topic=topic, msg_type=contacts_pb2.Contacts, callback=self.process_contact_message)
            self.contact_nodes.append(node) # Store node
            self.contact_subscriptions.append(sub) # Store subscription
            
            print(topic)

        self.started = True
    def stop_touch_listeners(self):

        # Clean up all contact sensor nodes and subscriptions
        for sub in self.contact_subscriptions:
            try:
                # In gz-transport13, unsubscribe is usually handled by the Node object's destruction
                # or if the subscription object has an unsubscribe method.
                # As of gz-transport13, there isn't a direct `sub.unsubscribe()`
                # Destroying the node explicitly often cleans up subscriptions.
                pass 
            except Exception as e:
                print(f"Error during contact subscription cleanup: {e}")
        self.contact_subscriptions = [] # Clear the list

        for node in self.contact_nodes:
            try:
                # Explicitly deleting the node or ensuring its reference count drops to 0
                # should trigger its destructor and clean up its resources.
                del node 
            except Exception as e:
                print(f"Error during contact node cleanup: {e}")
        self.contact_nodes = [] # Clear the list


    def close(self):
        """
        Cleans up all gz-transport nodes and subscriptions.
        """
        print("[*] SensorReading: Cleaning up lidar and contact sensor nodes...")
        self.stop_touch_listeners() # Stop contact listeners

        # Stop lidar listeners (from __init__)
        for sub in self.lidar_subscriptions:
            pass # No explicit unsubscribe method for gz-transport13 subscription object
        self.lidar_subscriptions = []

        for node in self.lidar_nodes:
            try:
                del node
            except Exception as e:
                print(f"Error during lidar node cleanup: {e}")
        self.lidar_nodes = []
        print("[*] SensorReading: All sensor nodes cleaned up.")
            
    
    def process_contact_message(self, msg):
        """
        Callback for processing messages from a raw contact sensor.s
        The message (msg) is of type gz.msgs.Contacts.
        """
        #print("Contact detected!")
        
        if self.started:
            if msg.contact:

                    #print(self.lidar_data)
                    
                    self.crashed = True

        

    def get_world_name(self):
        return self.world_name()
    
    
    