;; bam-radio decision engine.
;;
;; Copyright (c) 2019 Dennis Ogbe
;; Copyright (c) 2019 Tomohiro Arakawa

(in-package :bam-radio)

;; enable infix syntax
(named-readtables:in-readtable cmu-infix:syntax)

;;-----------------------------------------------------------------------------
;; housekeeping
;;-----------------------------------------------------------------------------

;; we keep a pointer to the C++ object around. use this pointer to call methods

(defvar *decision-engine-object* nil
  "The pointer to the decision engine C++ object.")

(defun set-decision-engine-ptr (ptr)
  (setf *decision-engine-object* ptr))

;; this table mirrors the subchannel table from sampling.h. it gives the
;; bandwidth in Hz for a given bandwith index.

(defvar *bandwidths* nil
  "A table mapping bandwidths to subchannels.")

(defun set-bandwidths (bws) (setf *bandwidths* bws))

(defvar *channelization* nil
  "A table mirroring the channelization table in channelization.cc.
A hash table mapping rf_bandwidth to a cons of max-bw-idx and a vector of
center frequencies, i.e. {5000000: (0 . #(-100000, -9000, 0, ... etc.)), ... etc.}")

(defun set-channelization-table (tbl) (setf *channelization* tbl))

(defvar *prev-inputs* nil
  "A list of all previous inputs.")

(defvar *prev-outputs* nil
  "A list of all previous outputs.")

(defvar +radio-sample-rate+ (floor 46.08e6)
  "The actual sample rate of the radio. Make sure to keep synced with bandwidth.cc.")

(defvar *my-network-id* nil
  "A cons cell containing my networks IP address. (<as integer> . <as string>)")

(defun set-network-id (id) (setf *my-network-id* id))

(defvar *enable-db-logging* t
  "This boolean enables of disables logging to the database.")

(defun enable-db-logging ()
  (setf *enable-db-logging* t))

(defun disable-db-logging ()
  (setf *enable-db-logging* nil))

(defun log-string-ext (&optional d s)
  (let ((logstr (format nil "DecisionEngine (~a): ~a~%" d s)))
    (handler-case (log-string logstr)
      (error (err)
        (format t "Fatal: Logging did not work: ~a. Bad things are happening.~%" err)))))

(defun init-random ()
  "Initialize the random number generator to a known state."
  (setf *random-state* +debug-random-state+))

(defvar *current-decision* nil
  "A list to push some information about the current decision to.")

(defvar *srnselect-1-wait-time* 4
  "The number of periodic steps to wait until the `srnselect-1' algorithm can
  move an srn again.")

(defvar *srnselect-1-wait-count* (make-hash-table)
  "A hash table SRN-ID => integer which keeps a count of the number of steps to
  wait until we can move an SRN again.")

(defun srnselect-1-decrease-counters (input)
  "Decrease the wait counter for all SRNs. Insert any new SRNs in the table."
  (loop for srn in (nodes input) do
       (let ((current-count (gethash (id srn) *srnselect-1-wait-count*)))
         (if (and current-count (> current-count 0))
             (decf (gethash (id srn) *srnselect-1-wait-count*))
             (setf (gethash (id srn) *srnselect-1-wait-count*) 0)))))

(defun srnselect-1-reset-counter (id)
  (setf (gethash id *srnselect-1-wait-count*)
        *srnselect-1-wait-time*))

(defmethod periodic-step-counters ((input decision-engine-input))
  "Dump all counter functions here. Called before any periodic step."
  (srnselect-1-decrease-counters input))

;; FIXME use macro for timed exec
(defun try-log-db-timed (lfun obj)
  (when *enable-db-logging*
    (handler-case
        (let ((tic) (toc) (serialize-time) (log-time)
              (bytes))
          (setf tic (get-time-now))
          (setf bytes (serialize-to-byte-vector obj))
          (setf toc (get-time-now))
          (setf serialize-time (msec (time- toc tic)))
          (setf tic (get-time-now))
          (unless (funcall lfun bytes)
            (log-string-ext "db-log" (format nil "C++ returned NIL.")))
          (setf toc (get-time-now))
          (setf log-time (msec (time- toc tic)))
          (log-string-ext "db-log" (format nil "serialize-time: ~ams log-time: ~ams"
                                           serialize-time log-time)))
      (error (err)
        (format t "Fatal: DB Logging did not work: ~a. Bad things are happening.~%" err)))))

(defun try-log-db (lfun obj)
  (when *enable-db-logging*
    (handler-case (funcall lfun (serialize-to-byte-vector obj))
      (error (err)
        (format t "Fatal: DB Logging did not work: ~a. Bad things are happening.~%" err)))))

#-sbcl
(defun try-log-db-threaded (lfun obj ext)
  "Spawn a new thread for every log attempt."
  (when *enable-db-logging*
    (handler-case
        (let ((tic) (toc) (exec-time)
              (name (format nil "db-log-~a-~a" ext (gensym))))
          (setf tic (get-time-now))
          (mp:process-run-function name
                                   #'(lambda ()
                                       ;; (log-string-ext "db-log" (format nil "Logging ~a..." name))
                                       (funcall lfun (serialize-to-byte-vector obj))
                                       ;; (log-string-ext "db-log" (format nil "Done logging ~a." name))
                                       (mp:exit-process)))
          (setf toc (get-time-now))
          (setf exec-time (msec (time- toc tic)))
          (log-string-ext "db-log" (format nil "~a log exec time: ~ams" name exec-time))))))

#+sbcl
(defun try-log-db-threaded (lfun obj ext)
  (declare (ignore ext))
  (try-log-db lfun obj))

(defmethod try-log-input ((input decision-engine-input))
  (try-log-db-threaded #'log-decision-engine-input input "in"))

(defmethod try-log-output ((output decision-engine-output))
  (try-log-db-threaded #'log-decision-engine-output output "out"))


;; TODO: there are currently no checks agains bad rf_bandwidth values anywhere
(defmethod get-true-offset ((node internal-node) (env environment))
  "Calculate the true offset in Hz based on the channel index and channel offset."
  (let ((base-cfreq (elt (cdr (gethash (rf-bandwidth env) *channelization*))
                         (chan-idx (tx-assignment node))))
        (cfreq-offset (chan-ofst (tx-assignment node))))
    (+ base-cfreq cfreq-offset)))

(defmethod channelization ((input decision-engine-input))
  (gethash (rf-bandwidth (env input)) *channelization*))

(defmethod channelization-cfreqs ((input decision-engine-input))
  (cdr (channelization input)))

(defmethod channelization-max-bw ((input decision-engine-input))
  (car (channelization input)))


;;-----------------------------------------------------------------------------
;; channel allocation
;;-----------------------------------------------------------------------------

(defmethod run-bandwidth-adaptation ((assgn tx-assignment-update) (input decision-engine-input))
  "Run the bandwidth adaptation process. Each node either increases or
  decreases their bandwidth if needed and possible."
  (let ((lowdc 0.4)
        (highdc 0.8)
        (max-bw-idx (or (car (gethash (rf-bandwidth (env input)) *channelization*)) 1))
        (min-bw-idx 1))
    ;; if a node's duty cycle is below the low threshold, attempt to decrease
    ;; its bandwidth. If it is above the high threshold, attempt to increase it
    (mapcar #'(lambda (srn)
                (let* ((srnid (id srn))
                       (dutycycle (est-duty-cycle srn))
                       (cur-bw-idx (bw-idx (gethash srnid (assignment-map assgn)))))
                  (when dutycycle ; dc could be nil, so we skip
                    (when (and (> dutycycle highdc) (< cur-bw-idx max-bw-idx))
                      ;; duty cyccle is high and there is room for more bandwidth
                      (setf (bw-idx (gethash srnid (assignment-map assgn)))
                            (1+ cur-bw-idx))
                      (setf (bandwidth-updated? assgn) t))
                    (when (and (< dutycycle lowdc) (> cur-bw-idx min-bw-idx))
                      ;; duty cycle is low and we can go lower
                      (setf (bw-idx (gethash srnid (assignment-map assgn)))
                            (1- cur-bw-idx))
                      (setf (bandwidth-updated? assgn) t)))))
            (nodes input))))

;; some random helper functions

(defun flip-biased-coin (bias)
  (< (random 1.0) bias))

(defun flip-fair-coin ()
  (flip-biased-coin 0.5))

(defun randomly-pick-one (seq)
  (let ((idx (random (length seq))))
    (elt seq idx)))

(defmethod random-channel-allocation ((assgn tx-assignment-update) (input decision-engine-input) fbcf rpof)
  "Proof of concept: decide whether to change channels with probability
  0.3. Then pick a random srn and change its assigned channel to a random
  unassigned channel."
  (let ((all-channels (cdr (gethash (rf-bandwidth (env input)) *channelization*))))
    (when (and (funcall fbcf 0.3) ; flip a 0.3-biased coin
               (not (= (length (nodes input))
                       (length all-channels))))
      ;; change a channel and modify the given tx-assignment
      (let* ((occupied-channels
              (mapcar #'(lambda (srn) (chan-idx (gethash (id srn) (assignment-map assgn))))
                      (nodes input)))
             (open-channels (set-difference (alexandria:iota (length all-channels))
                                            occupied-channels))
             (selected-srn-id (id (funcall rpof (nodes input))))
             (selected-channel (funcall rpof open-channels)))
        (setf (chan-idx (gethash selected-srn-id (assignment-map assgn)))
              selected-channel)
        (setf (channel-updated? assgn) t)))))

(defmethod process-env-update ((assgn tx-assignment-update) (input decision-engine-input))
  "Compute a new channel allocation after an environment update."
  (let ((new-bw (rf-bandwidth (env input))))
    ;; if there was a previous input, check if the rf_bandwidth changed. if it
    ;; didn't, we have to do nothing.
    (when *prev-inputs*
      (when (= (rf-bandwidth (env (car *prev-inputs*))) new-bw)
        (return-from process-env-update)))
    ;; else get the center frequencies and max bw for the new environment, clamp
    ;; any current bandwidths to the new max, and assign a new random center
    ;; frequency to each node.
    (push (list :process-env-update) *current-decision*)
    (let* ((new-max-bw (car (gethash new-bw *channelization*)))
           (new-center-freqs (cdr (gethash new-bw *channelization*)))
           (new-channel-idxs (alexandria:shuffle (alexandria:iota (length new-center-freqs)))))
      (mapcar #'(lambda (srn new-ch-idx)
                  (let ((current-bw-idx (bw-idx (gethash (id srn) (assignment-map assgn))))
                        (current-chan-idx (chan-idx (gethash (id srn) (assignment-map assgn)))))
                    (when (> current-bw-idx new-max-bw)
                      (setf (bw-idx (gethash (id srn) (assignment-map assgn))) new-max-bw)
                      (setf (bandwidth-updated? assgn) t)
                      (push (list :action :change-bandwidth :of (id srn)
                                  :from current-bw-idx
                                  :to new-max-bw)
                            *current-decision*))
                    (unless (= current-chan-idx new-ch-idx)
                      (setf (chan-idx (gethash (id srn) (assignment-map assgn)))
                            new-ch-idx)
                      (push (list :action :move (id srn)
                                  :from current-chan-idx
                                  :to new-ch-idx)
                            *current-decision*)
                      (setf (channel-updated? assgn) t))))
              (nodes input) new-channel-idxs))))

(defmethod rf-bandwidth-changed ((input decision-engine-input))
  "Has the rf-bandwidth of the environment changed?"
  (if *prev-inputs*
      ;; check whether changed w. r. t. previous step
      (unless (= (rf-bandwidth (env (car *prev-inputs*)))
                 (rf-bandwidth (env input)))
        t)
      ;; just reallocate...
      t))

(defmethod extract-tx-assignments ((input decision-engine-input))
  "Extract a fresh `tx-assignment-update' from the input. If a node does not
   have an assigned channel, we randomly assign one."
  (flet ((get-available-channels (input) ; FIXME consolidate with `get-open-channels'...
           (let ((all-channels (alexandria:iota (length (cdr (gethash (rf-bandwidth (env input))
                                                                      *channelization*)))))
                 (occupied-channels (mapcan #'(lambda (srn)
                                                (when (tx-assignment srn)
                                                  (list (chan-idx (tx-assignment srn)))))
                                            (nodes input))))
             (set-difference all-channels occupied-channels))))
    (let ((available-channels (get-available-channels input))
	  (max-bw-idx (or (car (gethash (rf-bandwidth (env input)) *channelization*)) 0))
          (ht (make-hash-table))
          (changed? nil))
      (mapcar #'(lambda (srn)
                  (setf (gethash (id srn) ht)
                          ;; either copy the current assignment or start with a
                          ;; random fresh assignment.
                          (if (tx-assignment srn)
                              (copy-instance (tx-assignment srn))
                              (let ((new-chan-idx (randomly-pick-one available-channels)))
                                (setf available-channels (delete new-chan-idx available-channels))
                                (setf changed? t)
                                (push (list :new-srn :id (id srn) :assign new-chan-idx) *current-decision*)
                                (make-instance 'transmit-assignment
                                               :bw-idx max-bw-idx
                                               :chan-idx new-chan-idx
                                               :chan-ofst 0
                                               :atten 0.0
                                               :silent nil)))))
              (nodes input))
      (make-instance 'tx-assignment-update
                     :assignment-map ht
                     :channel-updated? changed?
                     :bandwidth-updated? nil
                     :atten-updated? nil
                     :step-id (id input)))))

(defun band-to-chan-idx-range (band bw cfreqs)
  "Return a cons cell (START-IDX . END-IDX) indicating the range of channel
  indices from CFREQS the band BAND overlaps with when assuming that each
  channel's transmission is with bandwidth BW."
  (flet ((check-overlap (x1 x2 y1 y2) (and (<= x1 y2) (<= y1 x2))))
    (let* ((halfbw (/ bw 2))
           (overlap-channels
            (mapcan #'(lambda (cfreq idx)
                        (let ((lower #i(cfreq - halfbw))
                              (upper #i(cfreq + halfbw)))
                          (when (check-overlap (start band) (stop band) lower upper)
                            (list idx))))
                    cfreqs (alexandria:iota (length cfreqs)))))
      (when overlap-channels
        (cons (first overlap-channels) (alexandria:lastcar overlap-channels))))))

(defun freq-to-bin (freq full nbin)
  "Convert a freuquency to a bin index"
  (flet ((clip (num lo hi)
           (cond ((< num lo) lo)
                 ((> num hi) hi)
                 (t num)))
         (in-range (num lo hi)
           (cond ((< num lo) nil)
                 ((> num hi) nil)
                 (t t))))
    (let ((half #I(full/2)))
      (if (in-range freq #I(-half) half)
          #I( clip(floor(nbin * ((freq/full) + 0.5)), 0, nbin - 1) )
          (error "freq-to-bin: frequency ~a not in range [~a ~a]" freq #I(-half) half)))))

(defun thresh-psd-to-chan-idx (psd channelization full-bw)
  "Convert a thresholded PSD measurement into a list of channels (from the
   given channelization) in which energy was measured."
  (let* ((assumed-bandwidth (gethash (car channelization) *bandwidths*)) ; again assuming max-bandwidth
         (half (/ assumed-bandwidth 2))
         (npsd (length psd))
         (occ-chan))
    (flet ((lower (cfreq) (freq-to-bin (- cfreq half) full-bw npsd))
           (upper (cfreq) (freq-to-bin (+ cfreq half) full-bw npsd)))
      (map nil #'(lambda (cfreq idx)
                   (let* ((lb (lower cfreq))
                          (ub (upper cfreq))
                          (psd-bins-in-range (subseq psd lb (1+ ub))))
                     (when (find 1 psd-bins-in-range)
                       (push idx occ-chan))))
           (cdr channelization) (alexandria:iota (length (cdr channelization)))))
    occ-chan))

(defun chan-idx-list-to-binary (chan-idx nchan)
  "Convert a list of channel indices to a bit vector such that the bits indexed
   by the elements of the list are set to 1."
  (let ((out (make-array nchan :initial-element 0 :element-type 'bit)))
    (loop for idx in chan-idx do (setf (elt out idx) 1))
    out))

(defun spectrum-usage-to-interference-vector (spectrum-usage-list channelization scenario-cfreq)
  "Convert a list of `spectrum-usage' objects to a bit vector where a '1'
  indicates that the corresponding channel index is occupied."
  ;; use the maximum bandwidth when computing possible collisions. this might
  ;; need to change.
  (let* ((assumed-bandwidth (gethash (car channelization) *bandwidths*))
         (center-freqs (map 'list #'(lambda (cfreq) (+ cfreq scenario-cfreq))
                            (cdr channelization)))
         (ivec (make-array (length center-freqs) :initial-element 0 :element-type 'bit)))
    (when spectrum-usage-list
      (mapcar #'(lambda (usage)
                  (let ((occ-idxs (band-to-chan-idx-range (band usage) assumed-bandwidth center-freqs)))
                    (when occ-idxs
                      (let ((start-idx (car occ-idxs))
                            (stop-idx (cdr occ-idxs)))
                        (setf (subseq ivec start-idx (1+ stop-idx))
                              (make-array (1+ (- stop-idx start-idx))
                                          :initial-element 1 :element-type 'bit))))))
              spectrum-usage-list))
    ivec))

(defun spectrum-usage-to-interference-list (spectrum-usage-list channelization scenario-cfreq)
  (let* ((ivec (spectrum-usage-to-interference-vector spectrum-usage-list channelization scenario-cfreq))
         (olist))
    (map nil #'(lambda (bit idx)
                 (when (= bit 1)
                   (push idx olist)))
         ivec (alexandria:iota (length ivec)))
    (nreverse olist)))

(defmethod group-spectrum-usage ((input decision-engine-input))
  "Get hash table of all spectrum-usage objects sorted by team id,
   i.e. ((1 => (<spectrum-usage> <spectrum-usage> ...))
         (5 => (<spectrum-usage> <spectrum-usage> ...)) ...)"
  (let* ((tht (make-hash-table))
         (teams (loop for team in (collaborators input) collect
                     (progn (setf (gethash (id team) tht) (list))
                            (id team)))))
    (when (declared-spectrum input)
      (mapcar #'(lambda (su)
                  (loop for user in (users su) do
                       (when (find (network-id user) teams)
                         (push su (gethash (network-id user) tht)))))
              (declared-spectrum input)))
    tht))

(defmethod get-interference-tensor ((input decision-engine-input))
  "Return a hash table mapping team ids to interference maps for every internal
  node.

  Best explained by an example. If we have two collaborator networks, three
  nodes in our network, and the current channelization gives 4 discrete channel
  indices, we compute a map like this:

  {
     team 1: {
               node 1: #*1101
               node 2: #*1010
               node 3: #*0110
             }
     team 2: {
               node 1: #*1000
               node 2: #*1001
               node 3: #*1110
             }
  }

  So, for every collaborator and every one of our nodes we compute a bit vector
  the size of the current channelization, where a '1' indicates that this team
  is interfering with this node on this channel and a '0' indicates otherwise.
  This vector is computed from two data sources:
  (1) Each team's predicted spectrum-usage messages -- Clear.
  (2) Each node's thresholded PSD (if it exists)
      (After reading all predicted spectrum-usage messages and filling the map,
      we then check each node's PSD to see if a node measured any RF energy in
      the band that were declared to be in use by other teams. If our node did
      not measure energy, we mark this band as unoccupied.)"
  (let ((out (make-hash-table))
        (current-channelization (gethash (rf-bandwidth (env input)) *channelization*)))
    ;; (1): convert every team's spectrum usage messages to binary interference vectors
    (maphash #'(lambda (team-id su-msgs)
                 (let ((node-ht (make-hash-table))
                       (interference-vector (spectrum-usage-to-interference-vector
                                             su-msgs
                                             current-channelization
                                             (center-freq (env input)))))
                   (loop for node in (nodes input) do
                        (setf (gethash (id node) node-ht) interference-vector))
                   (setf (gethash team-id out) node-ht)))
             (group-spectrum-usage input))
    ;; (2): for every one of our nodes, try augmenting the spectrum-usage data
    ;; using the thresholded PSD values.
    (mapcar #'(lambda (srn)
                (let ((tpsd (thresh-psd srn))
                      (channels-with-energy))
                  (when tpsd
                    (setf channels-with-energy
                          (thresh-psd-to-chan-idx tpsd current-channelization +radio-sample-rate+))
                    (maphash #'(lambda (team-id node-ht)
                                 (declare (ignore team-id))
                                 (setf (gethash (id srn) node-ht)
                                       (bit-and (chan-idx-list-to-binary channels-with-energy
                                                                         (length (cdr current-channelization)))
                                                (gethash (id srn) node-ht))))
                             out))))
            (nodes input))
    ;; return the fresh interference tensor
    out))

(defmethod rank-scores ((input decision-engine-input))
  "Return a list of cons cells mapping team id to relative score (DARPA
  definition), sorted in descending order according to the score.

  Example: ((5 . 1.2) (3 . 0.9) (2 . 0.7))"
  (let* ((my-mps (loop for mandate in (mandates input)
		                sum (if (and (performance mandate)
				                         (> (scalar-performance (performance mandate)) 1))
			                      (point-value mandate) 0)))
	       (my-bonus-threshold (bonus-threshold (env input)))
	       (my-rel-progress (if (plusp my-bonus-threshold) (/ my-mps (bonus-threshold (env input))) 0))
	       (team-scores (list (cons (car *my-network-id*) my-rel-progress))))
    ;; loop through collaborators and add their scores to the list
    (loop for network in (collaborators input)
       do (when (and (reported-score network)
                     (scoring-point-threshold network)
                     (plusp (scoring-point-threshold network)))
            (push (cons (id network) (/ (reported-score network) (scoring-point-threshold network)))
                  team-scores)))
    ;; sort and done
    (setf team-scores (sort team-scores #'> :key #'cdr))
    team-scores))

(defmethod get-objective ((input decision-engine-input) sorted-scores)
  "Return one of {PROTECT-WORST, INCREASE-SCORE, INCREASE-BONUS-SCORE,
  HANDLE-INCUMBENT-VIOLATION, PROTECT-INCUMBENT-AND-PROTECT-WORST,
  PROTECT-INCUMBENT-AND-INCREASE-SCORE,
  PROTECT-INCUMBENT-AND-INCREASE-BONUS-SCORE} to give the decision
  engine an objective."
  (let ((my-score (cdr (assoc (car *my-network-id*) sorted-scores)))
        (incumbent-msg (and (incumbent input)
                            (last-message (incumbent input)))))
    ;; check the incumbent gate
    (if (and incumbent-msg
             (threshold-exceeded incumbent-msg))
        'HANDLE-INCUMBENT-VIOLATION     ;1) protect the incumbent
        ;; check if I'm above the bonus threshold
        (if (>= my-score 1)
            ;; all the other collaborators are also above the threshold?
            (if (every #'(lambda (collaborator-info) (>= (cdr collaborator-info) 1)) sorted-scores)
                (if incumbent-msg ;2) Everybody is above the threshold!
                    'PROTECT-INCUMBENT-AND-INCREASE-BONUS-SCORE
                    'INCREASE-BONUS-SCORE)
                (if incumbent-msg ;3) I'm performing well, but there's at least one team not doing well
                    'PROTECT-INCUMBENT-AND-PROTECT-WORST
                    'PROTECT-WORST))
            (if (every #'(lambda (collaborator-info) (>= (cdr collaborator-info) my-score)) sorted-scores)
                (if incumbent-msg ;4) I'm the least performing team and not achieving threshold
                    'PROTECT-INCUMBENT-AND-INCREASE-SCORE
                    'INCREASE-SCORE)
                (if incumbent-msg ;5) There's at least one team performing worse than us
                    'PROTECT-INCUMBENT-AND-PROTECT-WORST
                    'PROTECT-WORST))))))

(defmethod get-open-channels ((assgn tx-assignment-update) (input decision-engine-input))
  "Get all currently unoccupied channels."
  (let* ((all-channels (cdr (gethash (rf-bandwidth (env input)) *channelization*)))
         (occupied-channels (mapcar #'(lambda (srn)
                                        (chan-idx (gethash (id srn) (assignment-map assgn))))
                                    (nodes input)))
         (oc (set-difference (alexandria:iota (length all-channels)) occupied-channels)))
    oc))

(defun group-active-mandates-by-tx (input)
  "Return a map mapping srn ids to active flows offered to that srn."
  (let ((out (make-hash-table)))
    (mapcar #'(lambda (srn) (setf (gethash (id srn) out) '())) (nodes input))
    (mapcar #'(lambda (mandate)
                (when (and (active mandate) (rx mandate) (tx mandate))
                  (push mandate (gethash (tx mandate) out))))
            (mandates input))
    out))

;; lowest-sp-trend-least-collisions

(defun compute-sp-trend (active-mandates)
  "Compute a metric for the active mandates in ACTIVE-MANDATES. It is defined
  as follows:

           count of all active mandates with positive trend in scalar-performance
  metric = ----------------------------------------------------------------------
                              count of all active mandates

   The definition of 'positive trend' is:

  ((scalar-performance at this step) - (scalar-performance at previous step) > 0."
  (let ((pos-count 0)
        (prev-mandates (when *prev-inputs* (mandates (car *prev-inputs*)))))
    (loop for mandate in active-mandates do
         (when (performance mandate)
           (let* ((prev-mandate ; try to find this mandate id in the previous inputs
                   (and prev-mandates
                        (find mandate prev-mandates
                              :test #'(lambda (a b) (= (id a) (id b))))))
                  (delta (if (and prev-mandate (performance prev-mandate))
                             ;; compute difference
                             (- (scalar-performance (performance mandate))
                                (scalar-performance (performance prev-mandate)))
                             ;; else just take current value
                             (scalar-performance (performance mandate)))))
             (when (>= delta 0.0)
                 (incf pos-count)))))
    (/ pos-count (length active-mandates))))

(defun compute-sp-metric-positive (active-mandates)
  "Compute a metric for the active mandates in ACTIVE-MANDATES. It is defined
  as follows:

           count of all active mandates with positive scalar-performance
  metric = -------------------------------------------------------------
                          count of all active mandates
  "
  (let ((pos-count 0))
    (loop for mandate in active-mandates do
	       (when (and (performance mandate) (> (scalar-performance (performance mandate)) 0.0))
	         (incf pos-count)))
    (/ pos-count (length active-mandates))))

(defun compute-sp-metric-normalized-sp (active-mandates)
  "Compute a metric for the active mandates in ACTIVE-MANDATES. It is defined
  as follows:

           sum of all scalar-performance values of active mandates
  metric = -------------------------------------------------------
                     count of all active mandates
  "
  (let ((sp-sum 0))
    (loop for mandate in active-mandates do
	       (when (and (performance mandate) (> (scalar-performance (performance mandate)) 0.0))
	         (incf sp-sum (scalar-performance (performance mandate)))))
    (/ sp-sum (length active-mandates))))

(defun rank-by-active-mandate-metric (input compute-metric)
  "Return a list cons cells ((<srn-id> . <metric>) (<srn-id> . <metric>) ...),
  sorted in ascending order by the metric. The metric is computed by the
  function COMPUTE-METRIC, which is given a list of active mandates as an
  argument."
  (let ((out-ht (make-hash-table))
        (out-list)
        (grouped-mandates (group-active-mandates-by-tx input)))
    (maphash #'(lambda (srn-id active-mandates)
                 (if (= 0 (length active-mandates))
                     ;; remove from consideration if no active mandates
                     (remhash srn-id out-ht)
                     ;; compute metric and save to hash table
                     (setf (gethash srn-id out-ht)
                           (funcall compute-metric active-mandates))))
             grouped-mandates)
    (maphash #'(lambda (srn-id metric) (push (cons srn-id metric) out-list))
             out-ht)
    (setf out-list (sort out-list #'< :key #'cdr))
    out-list))

(defun hash-table-to-cons-list (ht)
  (let ((out))
    (maphash #'(lambda (key val)
                 (push (cons key val) out))
             ht)
    ;; sorting is to make tests not fail
    (sort out #'< :key #'car)))

(defun init-hash-table (keys init-val)
  (let ((out (make-hash-table)))
    (loop for key in keys do
         (setf (gethash key out) init-val))
    out))

(defun get-collision-count-subset (interf-tensor srn-id channel-subset)
  "See `get-collision-count', but with a subset of channels."
  (let ((out (init-hash-table channel-subset 0)))
    (maphash #'(lambda (team-id node-ht)
                 (declare (ignore team-id))
                 (let ((interf-vector (gethash srn-id node-ht)))
                   (loop for idx in channel-subset do
                        (when (= (elt interf-vector idx) 1)
                          (incf (gethash idx out))))))
             interf-tensor)
    (hash-table-to-cons-list out)))

(defun get-collision-count (interf-tensor srn-id nchan)
  "For srn with id SRN-ID, return a list of cons cells
   ((<chan-idx> . <count>) (<chan-idx> . <count>)), where <count> is the number
   of teams interfering on the corresponding <chan-idx> with SRN."
  (get-collision-count-subset interf-tensor srn-id
                              (alexandria:iota nchan)))

;; Mai's modified algorithm: weighted sum of receiver collisions

(defun compute-rel-points-weight (active-mandates)
  "Compute a weighting of receivers from a list of active mandates as follows:

               sum of point-values of all flows to receiver (i)
   weight(i) = ------------------------------------------------
                            sum of all point-values            "
  (let ((receiver-map (make-hash-table))
        (sum-points 0.0) (out))
    (loop for mandate in active-mandates do
         (incf sum-points (float (point-value mandate)))
         (if (gethash (rx mandate) receiver-map)
             (incf (gethash (rx mandate) receiver-map) (point-value mandate))
             (setf (gethash (rx mandate) receiver-map) (point-value mandate))))
    (maphash #'(lambda (rx-id rx-points)
                 (push (cons rx-id (float (/ rx-points sum-points))) out))
             receiver-map)
    out))

(defun get-rx-collision-metric-subset (interf-tensor srn-id channel-subset input compute-weights)
  "See `get-rx-collision-metric', but with a subset of channels."
  (let* ((active-mandates (gethash srn-id (group-active-mandates-by-tx input)))
         (rx-weights (funcall compute-weights active-mandates))
         (nchan (length channel-subset))
         (rx-collision-count (mapcar #'(lambda (sw) ; will get same order as rx_weights
                                         (get-collision-count-subset interf-tensor (car sw) channel-subset))
                                     rx-weights))
         (out (init-hash-table channel-subset 0.0)))
    (mapcar #'(lambda (rw rcc)
                (let ((weight (cdr rw)))
                  (loop for cc in rcc do
                       (let ((chan-idx (car cc))
                             (ncollisions (cdr cc)))
                         (incf (gethash chan-idx out)
                               (* weight ncollisions))))))
            rx-weights rx-collision-count)
    (hash-table-to-cons-list out)))

(defun get-rx-collision-metric (interf-tensor srn-id input compute-weights)
  "For srn with id SRN-ID, return a list of cons cells
   ((<chan-idx> . <metric>) (<chan-idx> . <metric>)), where <metric> is the
   weighted sum of all collision counts of all of the receivers of the active
   flows offered to srn with id SRN-ID.

   The parameter COMPUTE-WEIGHTS needs to be a function that, when given a list
   of active mandates, returns a list of cons cells ((<rx-id>
   . <weight>) (<rx-id> . <weight>) ...) where the 0 < <weight> < 1 is the
   weight given to each receiver when summing."
  (get-rx-collision-metric-subset interf-tensor
                                  srn-id
                                  (alexandria:iota (length (channelization-cfreqs input)))
                                  input
                                  compute-weights))

(defun get-rx-collisions-rel-point-weighted (interf-tensor srn-id input)
    (get-rx-collision-metric interf-tensor srn-id input #'compute-rel-points-weight))

(defun get-lowest-sp-trend-srn (input)
  "Return the SRN ID with the lowest normalized count of positive trending
  scalar-performance values. Return NIL if no active flows."
  (let ((ranked-srns (rank-by-active-mandate-metric input #'compute-sp-trend)))
    (when (> (length ranked-srns) 1)
      (caar ranked-srns))))

(defun lowest-sp-trend-least-collisions (assgn input interf-tensor sorted-scores tx?)
  "Attempt to increase our score by choosing the SRN with the lowest number of
   positive-trending scalar performance numbers and moving it to the channel
   with the lowest number of teams interfering."
  (declare (ignore sorted-scores)
           (special *current-decision*))
  (let ((selected-srn-id (get-lowest-sp-trend-srn input)))
    (if (not selected-srn-id)
        (push (list :no-action) *current-decision*)
        (funcall (if tx? #'move-to-lowest-tx-collisions #'move-to-lowest-rx-collisions)
                 selected-srn-id assgn input interf-tensor))))

;; -- end lowest-sp-trend-least-collisions

;; SRNSELECT-1: Choose the SRN to move by the following criteria:
;;
;; (0) if only one SRN has active flows, move it if its scalar performance is below 0.6
;; (1) filter out all SRNs that were moved during the last `*srnselect-1-wait-time*' steps
;; (2) filter out all SRNs with scalar performance above 0.6
;; (3) out of the remaining SRNs, choose the one with highest total of points attempted.
;;
;; Then move the selected srn using `move-to-lowest-rx-collisions'.

(defun srnselect-1-select-srn (input)
  ;; start with a list of all SRNs with active flows and a table mapping srns
  ;; to flows.
  (let* ((mandate-table (group-active-mandates-by-tx input))
         (srn-list (mapcan #'(lambda (srn-id)
                               (when (gethash srn-id mandate-table)
                                 (list srn-id)))
                           (alexandria:hash-table-keys mandate-table))))
    (flet ((avg-scalar-performance (srn-id)
             (let ((all-vals (mapcan #'(lambda (mandate)
                                                 (when (performance mandate)
                                                   (list (scalar-performance (performance mandate)))))
                                             (gethash srn-id mandate-table))))
               (if all-vals
                   (alexandria:mean all-vals)
                   0.0)))
           (total-points (srn-id) (reduce #'+ (gethash srn-id mandate-table)
                                          :key #'(lambda (mandate) (point-value mandate))))
           (debug-print (srn-list) (when t (log-string-ext "srnselect-1" (format nil "~A" srn-list)))))
      (debug-print srn-list)
      (if (<= (length srn-list) 1)
          ;; if there is only one srn, check and see whether its average scalar
          ;; performance is above 0.6. if yes, there is nothing to do.
          (when (and srn-list (> (avg-scalar-performance (car srn-list)) 0.6))
            (setf srn-list nil))
          (progn ;; else we continue to filter the list
            ;; remove all SRNs we are not supposed to touch
            (setf srn-list (remove-if #'(lambda (srn-id)
                                          (> (gethash srn-id *srnselect-1-wait-count*) 0))
                                      srn-list))
            (debug-print srn-list)
            ;; remove all SRNs with average scalar performance > 0.6
            (setf srn-list (remove-if #'(lambda (srn-id)
                                          (> (avg-scalar-performance srn-id) 0.6))
                                      srn-list))
            (debug-print srn-list)
            ;; remove all but the one srn with the highest total number of points attempted
            (setf srn-list (list (caar (sort (mapcar #'(lambda (id) (cons id (total-points id)))
                                                     srn-list)
                                             #'> :key #'cdr))))))
      ;; at this point there is either one or no srn id left in the list
      (debug-print srn-list)
      (when (> (length srn-list) 0)
        (let ((selected-srn-id (car srn-list)))
          (srnselect-1-reset-counter selected-srn-id)
          selected-srn-id)))))

(defun srnselect-1-channel-allocation (assgn input interf-tensor sorted-scores)
  "Select an srn using `srnselect-1-select-srn' and move it using
   `move-to-lowest-rx-collisions.'"
  (declare (ignore sorted-scores)
           (special *current-decision*))
  (let ((selected-srn-id (srnselect-1-select-srn input)))
    (if (not selected-srn-id)
        (push (list :no-action) *current-decision*)
        (move-to-lowest-rx-collisions selected-srn-id assgn input interf-tensor))))

;; -- end SRNSELECT-1

(defun move-to-lowest-collisions (srn-id assgn input ccf)
  "Move SRN-ID to a new channel. Obtain new channel by picking the minimum of
  the output of CCF."
  (declare (special *current-decision*))
  (let* ((selected-srn-id srn-id)
         (open-channels (get-open-channels assgn input))
         (collision-count (funcall ccf))
         ;; filter for only the unassigned channels, sort by collision count
         (collision-count-open-channels
          (sort (mapcan #'(lambda (cc) ; filter
                            (when (find (car cc) open-channels) (list cc)))
                        collision-count)
                #'< :key #'cdr)))
    ;; pick the one with the lowest collision count
    (let ((selected-channel (caar collision-count-open-channels))
          (prev-channel (chan-idx (gethash selected-srn-id (assignment-map assgn)))))
      ;; set it in the output
      (setf (chan-idx (gethash selected-srn-id (assignment-map assgn)))
            selected-channel)
      (setf (channel-updated? assgn)
            t)
      ;; some debug output
      (push (list :action :move selected-srn-id :from prev-channel :to selected-channel)
            *current-decision*))))

(defun move-to-lowest-rx-collisions (srn-id assgn input interf-tensor)
  "Move SRN-ID to a new channel. Obtain this new channel using a normalized
  collision count of the receiver's of SRN-ID's flows."
  (move-to-lowest-collisions
   srn-id assgn input
   #'(lambda () (get-rx-collisions-rel-point-weighted interf-tensor srn-id input))))

(defun move-to-lowest-tx-collisions (srn-id assgn input interf-tensor)
  "Move SRN-ID to a new channel. Obtain this new channel using the collision
  count of SRN-ID."
  (move-to-lowest-collisions
   srn-id assgn input
   #'(lambda () (get-collision-count interf-tensor srn-id
                                     (length (channelization-cfreqs input))))))

;; try to protect the worst team by moving out of its way in case we interfere with it

(defun move-away-from-worst (assgn input interf-tensor sorted-scores)
  "Attempt to deconflict from the worst-performing team by moving as many SRNs
   away from channels it occupies."
  (let* ((worst-team-id (car (alexandria:lastcar sorted-scores)))
         (current-channelization (gethash (rf-bandwidth (env input)) *channelization*))
         (worst-team-spectrum-usage (mapcan #'(lambda (su)
                                                (let ((netid (when (users su)
                                                               (when (car (users su))
                                                                 (network-id (car (users su)))))))
                                                  (when (= worst-team-id netid)
                                                    (list su))))
                                            (declared-spectrum input)))
         (worst-team-channel-idxs (spectrum-usage-to-interference-list worst-team-spectrum-usage
                                                                       current-channelization
                                                                       (center-freq (env input))))
         (all-channel-idxs (alexandria:iota (length (cdr current-channelization))))
         (open-channel-idxs (get-open-channels assgn input))
         (noconflict-channel-idxs (intersection (set-difference all-channel-idxs worst-team-channel-idxs)
                                                open-channel-idxs)))
    (maphash #'(lambda (srn-id tx-assignment)
                 (let ((this-srn-chan-idx (chan-idx tx-assignment)))
                   (when (and (find this-srn-chan-idx worst-team-channel-idxs)
                              (> (length noconflict-channel-idxs) 0))
                     ;; if this srn is on one of the team's channels, find the
                     ;; best remaining channel for it
                     (let* ((rx-collision-counts
                             (sort (get-rx-collision-metric-subset interf-tensor
                                                                    srn-id
                                                                    noconflict-channel-idxs
                                                                    input
                                                                    #'compute-rel-points-weight)
                                   #'< :key #'cdr))
                            (selected-chan-idx (caar rx-collision-counts)))
                       ;; set the output
                       (setf (chan-idx (gethash srn-id (assignment-map assgn)))
                             selected-chan-idx)
                       (setf (channel-updated? assgn) t)
                       ;; remove this channel from the available channels
                       (setf noconflict-channel-idxs (delete selected-chan-idx noconflict-channel-idxs))
                       ;; log that we have done something
                       (push (list :action :move srn-id :from this-srn-chan-idx :to selected-chan-idx)
                             *current-decision*)))))
             (assignment-map assgn))
    (unless (channel-updated? assgn)
      (push (list :no-action) *current-decision*))))

;; N.B. Remember that the bandwidth adaptation progress is driven independently
;; from this. Furthermore, we agree that when i.e. a chan-idx of ASSGN is
;; changed, we need to set the boolean (channel-changed? assgn) to t. (Same
;; with bandwidth and attenuation)

(defun chan-alloc-increase-score (assgn input interf-tensor sorted-scores)
  "Modify the `tx-assignment-update' ASSGN in a way that you believe increases
  our network's relative score. Assume that we are BELOW the bonus threshold."
  (srnselect-1-channel-allocation assgn input interf-tensor sorted-scores))

(defun chan-alloc-increase-bonus-score (assgn input interf-tensor sorted-scores)
  "Modify the `tx-assignment-update' ASSGN in a way that you believe increases
  our network's relative score. Assume that we are ABOVE the bonus threshold."
  ;; FIXME do something more appropriate
  (srnselect-1-channel-allocation assgn input interf-tensor sorted-scores))

(defun chan-alloc-protect-incumbent (assgn input interf-tensor sorted-scores)
  "Modify the `tx-assignment-update' ASSGN in a way that you believe protects
  the incumbent."
  (declare (ignore sorted-scores interf-tensor))
  (flet ((check-overlap (x1 x2 y1 y2) (and (<= x1 y2) (<= y1 x2)))
         (get-node-bandwidth (bw-idx) (gethash bw-idx *bandwidths*))
         (get-node-offset (chan-idx)
           (elt (cdr (gethash (rf-bandwidth (env input)) *channelization*)) chan-idx)))
    (when (incumbent input)
      (let* ((scenario-cfreq (center-freq (env input)))
             (inc-cf (+ scenario-cfreq (offset (incumbent input))))
             (inc-bw (bandwidth (incumbent input)))
             (inc-ub (+ inc-cf (/ inc-bw 2)))
             (inc-lb (- inc-cf (/ inc-bw 2))))
        (maphash #'(lambda (srnid assignment)
                     (declare (ignore srnid))
                     (if (let* ((node-cf (+ scenario-cfreq (get-node-offset (chan-idx assignment))))
                                (node-bw (get-node-bandwidth (bw-idx assignment)))
                                (node-ub (+ node-cf (/ node-bw 2)))
                                (node-lb (- node-cf (/ node-bw 2))))
                           (check-overlap inc-lb inc-ub node-lb node-ub))
                         (when (null (silent assignment))
                           (setf (silent assignment) t)
                           (setf (atten-updated? assgn) t))
                         (when (silent assignment)
                           (setf (silent assignment) nil)
                           (setf (atten-updated? assgn) t))))
                 (assignment-map assgn))))))

(defun chan-alloc-adjust-tx-power (assgn input interf-tensor sorted-scores)
  "Modify the `tx-assignment-update' ASSGN in a way that you believe protects
  the incumbent by adjusting the transmission power."
  (declare (ignore sorted-scores interf-tensor))
  (flet ((check-overlap (x1 x2 y1 y2) (and (<= x1 y2) (<= y1 x2)))
         (get-node-bandwidth (bw-idx) (gethash bw-idx *bandwidths*))
         (get-node-offset (chan-idx)
           (elt (cdr (gethash (rf-bandwidth (env input)) *channelization*)) chan-idx)))
    (when (and (incumbent input) (last-message (incumbent input)))
      (let* ((target-db -3)
             (atten-max 30)
             (backoff-rate 2)
             (recover-rate 0.5)
             (incumbent-message (last-message (incumbent input)))
             (current-level (if (eq 'PASSIVE-INCUMBENT (type-of (incumbent input)))
                                (power incumbent-message)
                                (inr incumbent-message)))
             (atten-diff (- current-level (+ (threshold incumbent-message) target-db)))
             (atten-adjustment (if (plusp atten-diff)
                                   (* backoff-rate atten-diff)
                                   (* recover-rate atten-diff)))
             (scenario-cfreq (center-freq (env input)))
             (inc-cf (+ scenario-cfreq (offset (incumbent input))))
             (inc-bw (bandwidth (incumbent input)))
             (inc-ub (+ inc-cf (/ inc-bw 2)))
             (inc-lb (- inc-cf (/ inc-bw 2))))
        (maphash #'(lambda (srnid assignment)
                     (declare (ignore srnid))
                     (if (let* ((node-cf (+ scenario-cfreq (get-node-offset (chan-idx assignment))))
                                (node-bw (get-node-bandwidth (bw-idx assignment)))
                                (node-ub (+ node-cf (/ node-bw 2)))
                                (node-lb (- node-cf (/ node-bw 2))))
                           (check-overlap inc-lb inc-ub node-lb node-ub))
                         (let* ((current-atten (atten assignment))
                                (new-atten (max 0 (min atten-max (+ atten-adjustment current-atten)))))
                           (when (/= current-atten new-atten)
                             (setf (atten assignment) new-atten)
                             (setf (atten-updated? assgn) t)))
                         (unless (zerop (atten assignment))
                           (setf (atten assignment) 0)
                           (setf (atten-updated? assgn) t))))
                 (assignment-map assgn))))))


(defun chan-alloc-protect-worst (assgn input interf-tensor sorted-scores)
  "Modify the `tx-assignment-update' ASSGN in a way that you believe increases
  the score of the worst performing team. Assume that this team is not us."
  (move-away-from-worst assgn input interf-tensor sorted-scores))

(defmethod open-channels-available ((input decision-engine-input))
  "If the number of channels in my channelization is the same as the number of
   nodes in my network, attempting to change a channel is pointless."
  (let ((nnodes (length (nodes input)))
        (nchan (length (cdr (gethash (rf-bandwidth (env input)) *channelization*)))))
    (< nnodes nchan)))

(defmethod run-channel-allocation ((assgn tx-assignment-update) (input decision-engine-input))
  "Run the channel allocation process."
  ;; only do channel allocation when there are open channels available.
  (when (open-channels-available input)
    ;; figure out scores, objectives, and interference
    (let* ((sorted-scores (rank-scores input))
           (objective (get-objective input sorted-scores))
           (interf-tensor (get-interference-tensor input)))
      (push (list :channel-allocation :objective objective)
            *current-decision*)
      ;; run the correct function
      (cond ((eq objective 'protect-worst)
             (chan-alloc-protect-worst assgn input interf-tensor sorted-scores))
            ((eq objective 'increase-score)
             (chan-alloc-increase-score assgn input interf-tensor sorted-scores))
            ((eq objective 'increase-bonus-score)
             (chan-alloc-increase-bonus-score assgn input interf-tensor sorted-scores))
            ((eq objective 'handle-incumbent-violation)
             (chan-alloc-adjust-tx-power assgn input interf-tensor sorted-scores))
            ((eq objective 'protect-incumbent-and-protect-worst)
             (chan-alloc-protect-worst assgn input interf-tensor sorted-scores)
             (chan-alloc-adjust-tx-power assgn input interf-tensor sorted-scores))
            ((eq objective 'protect-incumbent-and-increase-score)
             (chan-alloc-increase-score assgn input interf-tensor sorted-scores)
             (chan-alloc-adjust-tx-power assgn input interf-tensor sorted-scores))
            ((eq objective 'protect-incumbent-and-increase-bonus-score)
             (chan-alloc-increase-bonus-score assgn input interf-tensor sorted-scores)
             (chan-alloc-adjust-tx-power assgn input interf-tensor sorted-scores)))
      ;; return an output object using the information computed
      (make-instance 'decision-engine-output
                     :input-id (id input)
                     :assignment-update assgn
                     :objective objective
                     :sorted-scores sorted-scores
                     :interference-tensor interf-tensor
                     :decision (reverse *current-decision*)))))

(defmethod compute-cil-overlaps ((assgn tx-assignment-update) (input decision-engine-input))
  "Return a list of cons cell ((<id> . t/nil) (<id> . t/nil)...) for all SRN
  IDs that indicates whether an SRN is transmitting in a channel that a
  collaborator network has claimed using the CIL."
  (let ((all-claimed-channels))
    ;; collect a list of all claimed channels
    (maphash #'(lambda (team-id spectrum-usage-list)
                 (declare (ignore team-id))
                 (setf all-claimed-channels
                       (nconc (spectrum-usage-to-interference-list spectrum-usage-list
                                                                   (channelization input)
                                                                   (center-freq (env input)))
                              all-claimed-channels)))
             (group-spectrum-usage input))
    (setf all-claimed-channels (delete-duplicates all-claimed-channels))
    ;; construct the output
    (mapcar #'(lambda (srn)
                (cons (id srn)
                      (when (find (chan-idx (gethash (id srn) (assignment-map assgn)))
                                  all-claimed-channels)
                        t)))
            (nodes input))))

;;-----------------------------------------------------------------------------
;; The entry point for our C++ object
;;-----------------------------------------------------------------------------

(defun decision-engine-step (input)
  "The main entry point into our decision engine logic."
  ;; log to stdout / db
  (log-string-ext "step" (format nil "id: ~a trigger: ~a time: ~a"
                                 (id input) (trigger input) (time-stamp input)))
  (try-log-input input)
  (handler-case
      (let ((new-assignment (extract-tx-assignments input))
            (output-info) (*current-decision*))
        ;; preprocessing -- decrease counters
        (when (eql (trigger input) 'step)
          (periodic-step-counters input))
        ;; preprocessing -- adapt to environment
        (when (rf-bandwidth-changed input)
          (process-env-update new-assignment input))
        ;; preprocessing -- bandwidth adaptation
        (when (eql (trigger input) 'step)
          (run-bandwidth-adaptation new-assignment input))
        ;; final processing -- channel allocation
        (when (eql (trigger input) 'step)
          (setf output-info (run-channel-allocation new-assignment input)))
        ;; logging and housekeeping
        ;; transmit assignment update
        (when (or (channel-updated? new-assignment)
                  (bandwidth-updated? new-assignment)
                  (atten-updated? new-assignment))
          (update-tx-assignment *decision-engine-object* new-assignment))
        ;; overlap information update
        (let ((overlap-info (compute-cil-overlaps new-assignment input)))
          (when overlap-info
            (update-overlap-info *decision-engine-object* overlap-info)))
        ;; channel allocation output
        (when output-info
          (push output-info *prev-outputs*)
          (try-log-output output-info))
        ;; decision making chain
        (when *current-decision*
          (let ((*print-right-margin* most-positive-fixnum))
            (log-string-ext "decision-making" (format nil "~a" (reverse *current-decision*)))))
        ;; keep history
        (push input *prev-inputs*))
    (error (err)
      (if *rethrow-condition*
          (error "Error: ~a" err)
          (log-string-ext "error" (format nil "Caught condition: ~a Try again next time?" err))))))
