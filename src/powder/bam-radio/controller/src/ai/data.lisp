;; data.lisp
;;
;; data types for decision making
;;
;; Copyright (c) 2019 Dennis Ogbe

(in-package :bam-radio)

;; the classes below are intentionally POD-ish. we use them to pattern match
;; facts in the LISA production system

;; basics

(defclass location ()
  ((latitude  :type real :initarg :latitude :accessor latitude)
   (longitude :type real :initarg :longitude :accessor longitude)
   (elevation :type real :initarg :elevation :accessor elevation))
  (:documentation "A GPS Location."))

;; mandates/flows etc

(defclass mandate-performance ()
  ((mps :type integer :initarg :mps :accessor mps
        :documentation "number of measurement periods over which all PTs were met")
   (scalar-performance :type real :initarg :scalar-performance :accessor scalar-performance
                       :documentation "the scalar performance"))
  (:documentation "Performance of a mandate. See stats.h"))

(defclass mandate ()
  ((id :type integer :initarg :id :accessor id)
   (point-value :type integer :initarg :point-value :accessor point-value :documentation "point value of the flow")
   (hold-period :type integer :initarg :hold-period :accessor hold-period :documentation "mandated hold period")
   (active :type boolean :initarg :active :accessor active)
   (tx :type integer :initarg :tx :accessor tx)
   (rx :type integer :initarg :rx :accessor rx)
   (performance :type mandate-performance :initarg :performance :accessor performance))
  (:documentation "A mandate. Some call this a 'flow'."))

(defclass offered-traffic-rate ()
  ((src :type integer :initarg :src :accessor src)
   (dst :type integer :initarg :dst :accessor dst)
   (bps :type real :initarg :bps :accessor bps))
  (:documentation "Offered traffic stats"))

;; the spectrum

(defclass frequency-band ()
  ((start :type integer :initarg :start :accessor start
          :documentation "The lower of the two band edges in absolute frequency in Hz.")
   (stop :type integer :initarg :stop :accessor stop
         :documentation "The upper of the two band edges in absolute frequency in Hz."))
  (:documentation "A frequency band."))

(defclass spectrum-user ()
  ((kind :type symbol :initarg :kind :accessor kind)
   (network-id :type integer :initarg :network-id :accessor network-id
               :documentation "The network id of the user of a band")
   (tx-id :type integer :initarg :tx-id :accessor tx-id
          :documentation "The SRN id of the transmitter")
   (tx-pow-db :type real :initarg :tx-pow-db :accessor tx-pow-db
              :documentation "The declared transmit power of the user")))

(defclass spectrum-usage ()
  ((band :type frequency-band :initarg :band :accessor band)
   (users :type (list spectrum-user) :initarg :users :accessor users))
  (:documentation "A frequency band and a list of users."))

;; a radio node

(defclass node ()
  ((id :type integer :initarg :id :accessor id)
   (location :type location :initarg :location :accessor location))
  (:documentation "Any SRN (my or other networks)"))

;; our network

(defclass transmit-assignment ()
  ((bw-idx :type integer :initarg :bw-idx :accessor bw-idx :initform nil
           :documentation "index into *bandwidths* table")
   (chan-idx :type integer :initarg :chan-idx :accessor chan-idx  :initform nil
             :documentation "index into center frequency array from *channelization table*")
   (chan-ofst :type integer :initarg :chan-ofst :accessor chan-ofst  :initform 0
              :documentation "offset from channel center frequency")
   (atten :type real :initarg :atten :accessor atten :initform 0.0
          :documentation "attenuation in dB")
   (silent :type bool :initarg :silent :accessor silent :initform nil))
  (:documentation "A transmit assignment. Keep synced with discrete_channels.h"))

(defclass internal-node (node)
  ((tx-assignment :type transmit-assignment :initarg :tx-assignment :accessor tx-assignment :initform nil)
   (est-duty-cycle :type real :initarg :est-duty-cycle :accessor est-duty-cycle :initform nil)
   (real-psd :type (vector real *) :initarg :real-psd :accessor real-psd :initform nil)
   (thresh-psd :type (vector bit *) :initarg :thresh-psd :accessor thresh-psd :initform nil))
  (:documentation "A node in our network."))

;; other collaborators

(defclass network ()
  ((id :type integer :initarg :id :accessor id)
   (ip :type string :initarg :ip :accessor ip :documentation "this is the same as id, just human-readable.")
   (reported-score :type integer :initarg :reported-score :accessor reported-score)
   (scoring-point-threshold :type integer :initarg :scoring-point-threshold :accessor scoring-point-threshold)
   (nodes :type (list node) :initarg :nodes :accessor nodes))
  (:documentation "A collaborator network. This does not describe our network."))

(defclass incumbent ()
  ()
  (:documentation "An incumbent. Can be passive or active"))

(defclass passive-incumbent-message ()
  ((kind :type symbol :initarg :kind :accessor kind)
   (incumbent-id :type integer :initarg :incumbent-id :accessor incumbent-id)
   (report-time :type timestamp :initarg :report-time :accessor report-time)
   (power :type real :initarg :power :accessor power)
   (threshold :type real :initarg :threshold :accessor threshold)
   (offset :type integer :initarg :offset :accessor offset)
   (bandwidth :type integer :initarg :bandwidth :accessor bandwidth)
   (threshold-exceeded :type boolean :initarg :threshold-exceeded :accessor threshold-exceeded))
  (:documentation "The contents of the passive incumbent message."))

(defclass active-incumbent-message ()
  ((kind :type symbol :initarg :kind :accessor kind)
   (incumbent-id :type integer :initarg :incumbent-id :accessor incumbent-id)
   (report-time :type timestamp :initarg :report-time :accessor report-time)
   (inr :type real :initarg :inr :accessor inr)
   (threshold :type real :initarg :threshold :accessor threshold)
   (offset :type integer :initarg :offset :accessor offset)
   (bandwidth :type integer :initarg :bandwidth :accessor bandwidth)
   (threshold-exceeded :type boolean :initarg :threshold-exceeded :accessor threshold-exceeded))
  (:documentation "The contents of the active incumbent message."))

(defclass passive-incumbent (incumbent)
  ((offset :type integer :initarg :offset :accessor offset)
   (bandwidth :type integer :initarg :bandwidth :accessor bandwidth)
   (last-message :type passive-incument-message :initarg :last-message :initform nil
                 :accessor last-message
                 :documentation "The last message sent by the incumbent"))
  (:documentation "Information about passive incumbent. Supplied by colosseum."))

(defclass active-incumbent (incumbent)
  ((offset :type integer :initarg :offset :accessor offset)
   (bandwidth :type integer :initarg :bandwidth :accessor bandwidth)
   (last-message :type active-incument-message :initarg :last-message :initform nil
                 :accessor last-message
                 :documentation "The last message sent by the incumbent"))
  (:documentation "Information about active incumbent. Supplied by colosseum."))

;; the colosseum environment

(defclass environment ()
  ((collab-network-type :type symbol
                        :initarg :collab-network-type
                        :accessor collab-network-type
                        :documentation "One of INTERNET, SATCOM, HF, or UNSPEC")
   (rf-bandwidth :type integer :initarg :rf-bandwidth :accessor rf-bandwidth)
   (center-freq :type integer :initarg :center-freq :accessor center-freq)
   (bonus-threshold :type integer :initarg :bonus-threshold :accessor bonus-threshold)
   (has-incumbent :type boolean :initarg :has-incumbent :accessor has-incumbent :initform nil)
   (stage-number :type integer :initarg :stage-number :accessor stage-number))
  (:documentation "The colosseum environment."))

;; we pass all information to the AI code using one big object.

(defclass decision-engine-input ()
  ((trigger :type symbol
            :initarg :trigger
            :initform nil
            :accessor trigger
            :documentation "One of STEP, ENV-UPDATE, IM-UPDATE, INCUMBENT-NOTIFY.")
   (time-stamp :type timestamp
               :initarg :time-stamp
               :initform nil
               :accessor time-stamp
               :documentation "The time at which this input was produced.")
   (id :type integer
       :initarg :id
       :initform 0
       :accessor id
       :documentation "We keep a count.")
   (env :type environment
        :initarg :env
        :initform nil
        :accessor env
        :documentation "The current environment.")
   (mandates :type (list mandate)
             :initarg :mandates
             :initform nil
             :accessor mandates
             :documentation "A list of all current mandates.")
   (nodes :type (list internal-node)
          :initarg :nodes
          :initform nil
          :accessor nodes
          :documentation "A list of all nodes in our network.")
   (collaborators :type (list network)
                  :initarg :collaborators
                  :initform nil
                  :accessor collaborators
                  :documentation "A list of all collaborators.")
   (declared-spectrum :type (list spectrum-usage)
                      :initform nil
                      :initarg :declared-spectrum
                      :accessor declared-spectrum
		                  :documentation "Spectrum usage as declared by CIL collaborators.")
   (incumbent :type incumbent
              :initarg
              :incumbent
              :initform nil
              :accessor incumbent
              :documentation "Information about the incument, in case there is one")
   (offered-traffic-rates :type (list offered-traffic-rate)
                          :initarg :offered-traffic-rates
                          :initform nil
                          :accessor offered-traffic-rates))
  (:documentation "Input to the AI code."))

;; decision engine outputs

(defclass tx-assignment-update ()
  ((assignment-map :type hash-table :initarg :assignment-map :accessor assignment-map
                   :documentation "a map from SRN id to the new assignment")
   (channel-updated? :type bool :initarg :channel-updated? :accessor channel-updated?)
   (bandwidth-updated? :type bool :initarg :bandwidth-updated? :accessor bandwidth-updated?)
   (atten-updated? :type bool :initarg :atten-updated? :accessor atten-updated?)
   (step-id :type integer :initarg :step-id :accessor step-id :initform 0
            :documentation "the step id corresponding to this output."))
  (:documentation "A new transmit assignment was determinied. this data structure passes it on."))

(defmethod get-consed-assignment-map ((tau tx-assignment-update))
  "Turn the `assignment-map' slot into a list of cons cells for easier
  conversion."
  (loop for key being the hash-keys of (assignment-map tau)
     collect (cons key (gethash key (assignment-map tau)))))

(defclass decision-engine-output ()
  ((input-id :type integer :initarg :input-id :accessor input-id)
   (assignment-update :type tx-assignment-update :initarg :assignment-update :accessor assignment-update)
   (objective :type symbol :initarg :objective :accessor objective)
   (sorted-scores :type cons :initarg :sorted-scores :accessor sorted-scores)
   (interference-tensor :type hash-table :initarg :interference-tensor :accessor interference-tensor)
   (decision :type cons :initarg :decision :accessor decision)))

;; define some print methods for easier debugging

(defmethod print-object ((loc location) stream)
  (format stream "LOCATION {lat: ~a, lon: ~a, ele: ~a}"
          (latitude loc) (longitude loc) (elevation loc)))

(defmethod print-object ((rate offered-traffic-rate) stream)
  (format stream "OFFERED-TRAFFIC-RATE {src: ~a, dst: ~a, bps: ~a}"
	        (src rate) (dst rate) (bps rate)))

(defmethod print-object ((txass transmit-assignment) stream)
  (format stream "TX-ASSIGNMENT {bw-idx: ~a, chan-idx: ~a, chan-ofst: ~a, atten: ~a, silent: ~a}"
          (bw-idx txass) (chan-idx txass) (chan-ofst txass) (atten txass) (silent txass)))

(defmethod print-object ((tau tx-assignment-update) stream)
  (format stream "TX-ASSIGNMENT-UPDATE step id: ~a chan changed? ~a bw changed? ~a atten changed? ~a~%"
          (step-id tau) (channel-updated? tau) (bandwidth-updated? tau) (atten-updated? tau))
  (print-nested-hash-table (assignment-map tau) stream))


;; stack overflow to the rescue:
;; https://stackoverflow.com/questions/11067899
#+ecl
(defgeneric copy-instance (object &rest initargs &key &allow-other-keys)
  (:documentation "Makes and returns a shallow copy of OBJECT.

  An uninitialized object of the same class as OBJECT is allocated by
  calling ALLOCATE-INSTANCE.  For all slots returned by
  CLASS-SLOTS, the returned object has the
  same slot values and slot-unbound status as OBJECT.

  REINITIALIZE-INSTANCE is called to update the copy with INITARGS.")
  (:method ((object standard-object) &rest initargs &key &allow-other-keys)
    (let* ((class (class-of object))
           (copy (allocate-instance class)))
      (dolist (slot-name (mapcar #'mop:slot-definition-name (mop:class-slots class)))
        (when (slot-boundp object slot-name)
          (setf (slot-value copy slot-name)
            (slot-value object slot-name))))
      (apply #'reinitialize-instance copy initargs))))

#+sbcl
(defgeneric copy-instance (object &rest initargs &key &allow-other-keys)
  (:documentation "Makes and returns a shallow copy of OBJECT.

  An uninitialized object of the same class as OBJECT is allocated by
  calling ALLOCATE-INSTANCE.  For all slots returned by
  CLASS-SLOTS, the returned object has the
  same slot values and slot-unbound status as OBJECT.

  REINITIALIZE-INSTANCE is called to update the copy with INITARGS.")
  (:method ((object standard-object) &rest initargs &key &allow-other-keys)
    (let* ((class (class-of object))
           (copy (allocate-instance class)))
      (dolist (slot-name (mapcar #'sb-mop:slot-definition-name (sb-mop:class-slots class)))
        (when (slot-boundp object slot-name)
          (setf (slot-value copy slot-name)
            (slot-value object slot-name))))
      (apply #'reinitialize-instance copy initargs))))
