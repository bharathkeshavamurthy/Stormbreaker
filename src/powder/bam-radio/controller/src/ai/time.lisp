;; deal with time in a manner similar to std::chrono
;;
;; Copyright (c) 2019 Dennis Ogbe

(in-package :bam-radio)

;; we use the timestamp in a similar manner as the chrono::time_point in C++. we extend
;; it with a nsec "duration" class that acts like a chrono::duration

(defclass duration nil
  ((nsec :type integer :initarg :nsec :accessor nsec))
  (:documentation "A duration in nanoseconds."))

(defmethod msec ((dur duration))
  (/ (nsec dur) 1e6))

;; FIXME wtf is this?
;; (defmacro make-duration (n &optional unit)
;;   `(let ((ns (cond ((eql ,unit :hour) (* (* ,n 3600) 1000000000))
;;                    ((eql ,unit :min) (* (* ,n 60) 1000000000))
;;                    ((eql ,unit :sec) (* ,n 1000000000))
;;                    ((eql ,unit :usec) (* ,n 1000000))
;;                    ((eql ,unit :msec) (* ,n 1000))
;;                    ((eql ,unit :nsec) ,n)
;;                    (t n))))
;;      (make-instance 'duration :nsec ns)))

(defmethod print-object ((d duration) stream)
  (format stream "~Ans" (nsec d)))

(defmethod time+ ((tp timestamp) (d duration))
  (timestamp+ tp (nsec d) :nsec))

(defmethod time- ((tp timestamp) (d duration))
  (timestamp- tp (nsec d) :nsec))

(defmethod time- ((t2 timestamp) (t1 timestamp))
  "Not the best implementation of this..."
  (when (timestamp< t2 t1) (error "Cannot subtract these time stamps"))
  (if (timestamp= t1 t2)
      (make-instance 'duration :nsec 0)
      (let ((full1 (+ (* 1000000000 (sec-of t1)) (nsec-of t1)) )
            (full2 (+ (* 1000000000 (sec-of t2)) (nsec-of t2)) ))
        (make-instance 'duration :nsec (- full2 full1)))))
