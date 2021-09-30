import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Brando01 OpenCV tracker
struct Brando02: ParsableCommand {
    func run() {

        let np = Python.import("numpy")
        let cv2 = Python.import("cv2")
        let os = Python.import("os")
        // let imutils = Python.import("utils")
        print(Python.version)
        print(Python.tuple([1,3,4]))
        let image_names = os.listdir("../OIST_Data/downsampled")
		let track_names = os.listdir("../OIST_Data/tracks")
        image_names.sort()
		track_names.sort()
        // let tracker = cv2.TrackerCSRT_create()
        let track = track_names[10]
		let frame = cv2.imread("../OIST_Data/downsampled/" + image_names[0])
        let centers = Python.list()
		let fs = Python.open("../OIST_Data/tracks/" + track, "r")
		let lines = fs.readlines()
        print(type(of: lines))
        var i = 0
		for line in lines {
            if i == 0 {
                i += 1
                continue
            }
            i += 1
            // print(type(of: line))
            let lineSwift = String(line)
            // print(type(of: lineSwift))
            
            let lineSwift2 = lineSwift ?? ""
            // print(lineSwift2)
            let nums = lineSwift2.components(separatedBy: " ")
            // print(nums)
            let height = Float(nums[1])
            let width = Float(nums[0])
			centers.append(Python.tuple([Python.float(width),Python.float(height)]))
        }
        // print(centers)


        let width1 = Float(centers[0][0])
		let height1 = Float(centers[0][1])
        let width = width1 ?? 0
        let height = height1 ?? 0
		let BB = Python.tuple([Int(width-35),Int(height-35),70,70])
        let tracker = cv2.TrackerMIL_create()
        // print(frames.first!.makeNumpyArray())
            // BB = (width-35,height-35,70,70)
        print(type(of: tracker))
        tracker[dynamicMember: "init"](frame, BB)
        var results = [PythonObject]()
        for image_name in image_names {
			let framei = cv2.imread("../OIST_Data/downsampled/" + image_name)
            var a = tracker[dynamicMember: "update"](framei).tuple2
            let track_success = a.0
            let newBB = a.1
            if Bool(track_success)! {
                results.append(newBB)
            }
            // if Bool(track_success) {
            //     results.append(BB)
            // }
        }

    
    }

}