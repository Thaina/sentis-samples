package com.unity.opencvutils;

import org.opencv.core.*;
import org.opencv.calib3d.Calib3d;

public class Proxy
{
    public static double[] solvePnP(double[] points3D, double[] points2D, double[] cameraProjectionMatrix, double imgWidth, double imgHeight, boolean useExtrinsicGuess, int solvePnPFlags) {
        try {
            // Convert 3D points to MatOfPoint3f
            MatOfPoint3f objectPoints = new MatOfPoint3f();
            int numPoints = points3D.length / 3;
            Point3[] points3DArr = new Point3[numPoints];
            for (int i = 0; i < numPoints; i++) {
                points3DArr[i] = new Point3(points3D[i * 3], points3D[i * 3 + 1], points3D[i * 3 + 2]);
            }
            objectPoints.fromArray(points3DArr);

            // Convert 2D points to MatOfPoint2f
            MatOfPoint2f imagePoints = new MatOfPoint2f();
            Point[] points2DArr = new Point[numPoints];
            for (int i = 0; i < numPoints; i++) {
                points2DArr[i] = new Point(points2D[i * 2], points2D[i * 2 + 1]);
            }
            imagePoints.fromArray(points2DArr);

            // Create camera matrix from projection matrix
            Mat projMatrix = new Mat(4, 4, CvType.CV_64F);
            projMatrix.put(0, 0, cameraProjectionMatrix);

            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
            // Extract camera matrix components
            cameraMatrix.put(0, 0, cameraProjectionMatrix[0] * imgWidth / 2.0); // fx
            cameraMatrix.put(1, 1, cameraProjectionMatrix[5] * imgHeight / 2.0); // fy
            cameraMatrix.put(0, 2, imgWidth / 2.0); // cx
            cameraMatrix.put(1, 2, imgHeight / 2.0); // cy
            cameraMatrix.put(2, 2, 1.0);

            // Create distortion coefficients (assume no distortion)
            MatOfDouble distCoeffs = new MatOfDouble(0.1, // k1 - radial distortion
                                                    0.2, // k2
                                                    0.01, // p1 - tangential distortion
                                                    0.01, // p2
                                                    0.3  // k3
                                                    );

            // Output rotation and translation
            Mat rvec = new Mat();
            Mat tvec = new Mat();

            Calib3d.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, solvePnPFlags);

            Mat rotMat = new Mat();
            Calib3d.Rodrigues(rvec, rotMat);

            // Create 4x4 transformation matrix
            Mat transformMat = Mat.zeros(4, 4, CvType.CV_64F);
            // Copy rotation matrix
            for(int i = 0; i < 3; i++) {
                for(int j = 0; j < 3; j++) {
                    transformMat.put(i, j, rotMat.get(i, j)[0]);
                }
            }
            // Copy translation vector
            transformMat.put(0, 3, tvec.get(0, 0)[0]);
            transformMat.put(1, 3, tvec.get(1, 0)[0]);
            transformMat.put(2, 3, tvec.get(2, 0)[0]);
            // Set bottom row to [0,0,0,1]
            transformMat.put(3, 3, 1.0);

            // Convert to double array
            double[] transformMatrix = new double[16];
            transformMat.get(0, 0, transformMatrix);
            return transformMatrix;
        } catch (Exception e) {
            e.printStackTrace();
            return new double[16]; // Return empty array on error
        }
    }
}
