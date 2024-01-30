#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:24:12 2023

@author: tjards
"""


class HerdAndShepherds:
    def __init__(self, shepherd_name):
        self.shepherd_name = shepherd_name
        self.state_herd_i = "Some state"
        self.herd = self.Herd()

    class Herd:
        def __init__(self):
            # Accessing the outer class attribute
            outer_class_attribute = HerdAndShepherds.shepherd_name
            print(f"Accessing outer class attribute from inner class: {outer_class_attribute}")

            # Accessing the outer class attribute state_herd_i
            outer_class_state = HerdAndShepherds.state_herd_i
            print(f"Accessing outer class state_herd_i from inner class: {outer_class_state}")

# Example usage
herd_and_shepherds_instance = HerdAndShepherds("John")