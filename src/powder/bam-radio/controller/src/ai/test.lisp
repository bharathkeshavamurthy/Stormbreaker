;; bam-radio tests.
;;
;; Copyright (c) 2019 Dennis Ogbe

(in-package :cl-user)

(defpackage :bam-radio-tests
  (:use :common-lisp
        :local-time
        :bam-radio
        :parachute))

(in-package :bam-radio-tests)

(defvar *verbose-output* nil
  "if T, print extra information to stdout.")

(define-test bam-radio
  "The main test. run this to test everything.")

;;-----------------------------------------------------------------------------
;; channel allocation code
;;-----------------------------------------------------------------------------

(define-test bandwidth-adaptation
  :parent bam-radio
  (defun test-bw-adapt-single-srn (in-duty-cycle in-bw-idx expected-bw-idx)
    (let* ((br::*channelization* br::+debug-channelization-table+)
           ;; fake decision engine input
           (fake-input (make-instance 'br::decision-engine-input
                                      :env (make-instance 'br::environment :rf-bandwidth (floor 40e6))
                                      :nodes `(,(make-instance 'br::internal-node
                                                               :id 0
                                                               :est-duty-cycle in-duty-cycle
                                                               :tx-assignment (make-instance 'br::transmit-assignment
                                                                                             :bw-idx in-bw-idx)))))
           (assgn (br::extract-tx-assignments fake-input)))
      ;; run the function
      (br::run-bandwidth-adaptation assgn fake-input)
      ;; compare
      (let ((my-bw-idx (br::bw-idx (gethash (br::id (car (br::nodes fake-input)))
                                            (br::assignment-map assgn))))
            (my-bw-changed (br::bandwidth-updated? assgn))
            (expected-bw-changed (not (= in-bw-idx expected-bw-idx))))
        (is = expected-bw-idx my-bw-idx)
        (is eq expected-bw-changed my-bw-changed)))))

(define-test (bandwidth-adaptation below-thresh-1)
  "Test the bandwidth adaptation process with one SRN below the lower threshold."
  (test-bw-adapt-single-srn 0.3 1 1))

(define-test (bandwidth-adaptation below-thresh-2)
  "Test the bandwidth adaptation process with one SRN below the lower threshold."
  (test-bw-adapt-single-srn 0.3 2 1))

(define-test (bandwidth-adaptation above-thresh-1)
  "Test the bandwidth adaptation process with one SRN above the higher threshold."
  (test-bw-adapt-single-srn 0.85 4 4))

(define-test (bandwidth-adaptation above-thresh-2)
  "Test the bandwidth adaptation process with one SRN above the higher threshold."
  (test-bw-adapt-single-srn 0.85 2 3))

(define-test (bandwidth-adaptation within-thresh)
  "Test the bandwidth adaptation process with one SRN within the threshold."
  (test-bw-adapt-single-srn 0.5 3 3))

(define-test (bandwidth-adaptation no-duty-cycle)
  "Test the bandwidth adaptation process when no duty cycle is given."
  (test-bw-adapt-single-srn nil 1 1))

(define-test random-channel-allocation
  :parent bam-radio
  ;; undo randomness
  (defun always-heads (bias)
    T)
  (defun always-pick (seq idx)
    (elt seq idx)))

(define-test (random-channel-allocation basic)
  ;; set-up
  (let* ((br::*channelization* br::+debug-channelization-table+)
         (fake-input (make-instance 'br::decision-engine-input
                                    :env (make-instance 'br::environment :rf-bandwidth (floor 40e6))
                                    :nodes `(,(make-instance 'br::internal-node
                                                             :id 0
                                                             :est-duty-cycle 0.5 ;; no changes here
                                                             :tx-assignment (make-instance 'br::transmit-assignment
                                                                                           :bw-idx 1
                                                                                           :chan-idx 0)))))
         (assgn (br::extract-tx-assignments fake-input)))
    ;; run the function (without randomness)
    (br::random-channel-allocation assgn fake-input #'always-heads #'(lambda (seq) (always-pick seq 0)))
    ;; compare
    (let ((expected-chan-idx 1)
          (my-chan-idx (br::chan-idx (gethash (br::id (car (br::nodes fake-input)))
                                              (br::assignment-map assgn)))))
      (is = expected-chan-idx my-chan-idx))))

(define-test chan-alloc-misc
    :parent bam-radio)

(define-test (chan-alloc-misc f2bin)
  ;; same as above
  (let ((f 80) (n 8))
    (mapcar #'(lambda (b r) (is equalp b (bam-radio::freq-to-bin r f n)))
            (alexandria:iota 8)
            (alexandria:iota 8 :start -35 :step 10))))

(define-test (chan-alloc-misc get-open-channels)
  ;; TODO
  )

(define-test (chan-alloc-misc group-active-mandates-by-tx)
  ;; TODO
  )

(define-test (chan-alloc-misc extract-tx-assignments)
  (let* ((nodes (list (make-instance 'br::internal-node
                                     :id 33
                                     :tx-assignment (make-instance 'br::transmit-assignment
                                                                   :bw-idx 0
                                                                   :chan-idx 4))
                      (make-instance 'br::internal-node
                                     :id 56
                                     :tx-assignment (make-instance 'br::transmit-assignment
                                                                   :bw-idx 0
                                                                   :chan-idx 6))
                      (make-instance 'br::internal-node
                                     :id 66
                                     :tx-assignment nil)))
         (env (make-instance 'br::environment :rf-bandwidth (floor 8e6)))
         (fake-input (make-instance 'br::decision-engine-input
                                    :nodes nodes :env env))
         (br::*channelization* br::+debug-channelization-table+)
         (my-output)
         (out-ch-idx))
    (setf my-output (br::extract-tx-assignments fake-input))
    (setf out-ch-idx (br::chan-idx (gethash 66 (br::assignment-map my-output))))
    (when *verbose-output*
      (format t "out-ch-idx: ~a~%" out-ch-idx))
    (is eq t (and (not (= out-ch-idx 4))
                  (not (= out-ch-idx 6))))))

(define-test (chan-alloc-misc oc-avail)
  (defun test-oc-avail (rfb nnodes expected)
    (let* ((br::*channelization* br::+debug-channelization-table+)
           (env (make-instance 'br::environment
                               :rf-bandwidth (floor rfb)))
           (nodes (loop for k from 1 to nnodes collect
                       (make-instance 'br::internal-node)))
           (fake-input (make-instance 'br::decision-engine-input
                                      :env env :nodes nodes)))
      (is eq expected (br::open-channels-available fake-input))))
  (test-oc-avail 40e6 10 t)
  (test-oc-avail 10e6 10 nil)
  (test-oc-avail 20e6 23 nil))

(define-test interf-tensor
  :parent bam-radio
  "Unit tests for the construction of interference tensors.")

(define-test (interf-tensor group-su)
  "Test the grouping of spectrum usage messages `group-spectrum-usage'."
  (defun dumb-su-eq (a b)
    (when (and (equal (br::start (br::band a))
                      (br::start (br::band b)))
               (equal (br::stop (br::band a))
                      (br::stop (br::band b)))
               (equal (br::network-id (car (br::users a)))
                      (br::network-id (car (br::users b)))))
      t))
  ;; set-up: two teams, each with one spectrum usage declaration
  (let* ((declspec (list (make-instance 'br::spectrum-usage
                                        :band (make-instance 'br::frequency-band
                                                             :start 1000
                                                             :stop 2000)
                                        :users (list (make-instance 'br::spectrum-user
                                                                    :kind 'network
                                                                    :network-id 5)))
                         (make-instance 'br::spectrum-usage
                                        :band (make-instance 'br::frequency-band
                                                             :start 3000
                                                             :stop 4000)
                                        :users (list (make-instance 'br::spectrum-user
                                                                    :kind 'network
                                                                    :network-id 99)))))
         (teams (list (make-instance 'br::network :id 5)
                      (make-instance 'br::network :id 99)))
         (fake-input (make-instance 'br::decision-engine-input
                                       :declared-spectrum declspec
                                       :collaborators teams))
         (expected-output (let ((ht (make-hash-table)))
                            (setf (gethash 5 ht) (car declspec))
                            (setf (gethash 99 ht) (cadr declspec))
                            ht))
         (my-output (br::group-spectrum-usage fake-input)))
    (maphash #'(lambda (key val)
                 (is dumb-su-eq val (car (gethash key my-output))))
               expected-output)))

(define-test (interf-tensor idx2bin)
  (let* ((nchan 8)
         (my-chan '(1 3 4 6))
         (my-out (br::chan-idx-list-to-binary my-chan nchan))
         (expected #*01011010))
    (is equalp expected my-out)))

(define-test (interf-tensor band2chanidxr)
  (let* ((current-channelization (gethash 8000000 br::+debug-channelization-table+))
         (lower-upper (list '(-2861647 . -2185412)
                            '(-2085176 . -1797176)
                            '(244235 . 920471)
                            '(1797176 . 2861647)))
         (fo 1235) ; Hz if we want to push some stuff around
         (cfreq 1000000000) ; 1 GHz scenario center frequency
         ;; (1) a band starting slightly before 1 and ending inside of 2
         ;; (2) a band starting and ending in 3
         ;; (3) a band starting inside of 9 and ending slightly outside of 4
         ;; (4) a band starting before 13 and ending after 15
         (interfering-bands (mapcar #'(lambda (lu ff)
                                        (let ((lower (+ (car lu) (car ff)))
                                              (upper (+ (cdr lu) (cdr ff))))
                                          (make-instance 'br::frequency-band
                                                         :start (+ lower cfreq)
                                                         :stop (+ upper cfreq))))
                                    lower-upper
                                    (list (cons (- fo) (- fo))
                                          (cons fo (- fo))
                                          (cons fo fo)
                                          (cons (- fo) fo))))
         (expected-outputs (list '(1 . 2)
                                 '(3 . 3)
                                 '(9 . 10)
                                 '(13 . 15))))
    (mapcar #'(lambda (band expected-output)
                (let ((my-out (br::band-to-chan-idx-range band
                                                          (gethash (car current-channelization)
                                                                   br::+debug-bandwidth-table+)
                                                          (map 'list #'(lambda (cf) (+ cf cfreq))
                                                               (cdr current-channelization)))))
                  ;; (format t "expected: ~a~%my      : ~a~%" expected-output my-out)
                  (is equalp expected-output my-out)))
            interfering-bands expected-outputs)))

(define-test (interf-tensor su2iv)
  :depends-on (idx2bin band2chanidxr)
  "Test the function `spectrum-usage-to-interference-vector'"
  ;; set-up: 8MHz environment, two spectrum-usage messages that take out
  ;; channels 1, 2, 3, 9, 10, 13, 14, 15. boundaries computed by hand from
  ;; *channelization* table and assuming 288kHz bandwidth
  (let* ((current-channelization (gethash 8000000 br::+debug-channelization-table+))
         (lower-upper (list '(-2861647 . -2185412)
                            '(-2085176 . -1797176)
                            '(244235 . 920471)
                            '(1797176 . 2861647)))
         (target-channels (list '(1 2)
                                '(3)
                                '(9 10)
                                '(13 14 15)))
         (fo 1235) ; Hz if we want to push some stuff around
         (cfreq 1000000000) ; 1 GHz scenario center frequency
         ;; (1) a band starting slightly before 1 and ending inside of 2
         ;; (2) a band starting and ending in 3
         ;; (3) a band starting inside of 9 and ending slightly outside of 4
         ;; (4) a band starting before 13 and ending after 15
         (interfering-bands (mapcar #'(lambda (lu ff)
                                        (let ((lower (+ (car lu) (car ff)))
                                              (upper (+ (cdr lu) (cdr ff))))
                                          (make-instance 'br::frequency-band
                                                         :start (+ lower cfreq)
                                                         :stop (+ upper cfreq))))
                                    lower-upper
                                    (list (cons (- fo) (- fo))
                                          (cons fo (- fo))
                                          (cons fo fo)
                                          (cons (- fo) fo))))
         (expected-output (br::chan-idx-list-to-binary
                           (apply #'append target-channels)
                           (length (cdr current-channelization))))
         (my-output (br::spectrum-usage-to-interference-vector
                     (mapcar #'(lambda (band)
                                 (make-instance 'br::spectrum-usage
                                                :band band))
                             interfering-bands)
                     current-channelization
                     cfreq)))
    (is equalp expected-output my-output)))

(define-test (interf-tensor tpsd2chidx)
  :depends-on (f2bin)
  "Test the function `thresh-psd-to-chan-idx'."
  ;; set-up: 20MHz scenario, 1024-point FFT, we artificially create a PSD that
  ;; knocks out channels 2 through 5 assuming the max. bandwidth.
  (let* ((fudge 1342)
         (nbin 1024)
         (current-channelization (gethash 20000000 br::+debug-channelization-table+))
         (halfbw (/ (gethash (car current-channelization) br::+debug-bandwidth-table+) 2))
         (lower-freq (- (elt (cdr current-channelization) 2) halfbw fudge))
         (upper-freq (+ (elt (cdr current-channelization) 5) halfbw fudge))
         (lower-bin (br::freq-to-bin lower-freq br::+radio-sample-rate+ nbin))
         (upper-bin (br::freq-to-bin upper-freq br::+radio-sample-rate+ nbin))
         (psd (let ((tpsd (make-array nbin :initial-element 0 :element-type 'bit))
                    (ones (make-array (1+ (- upper-bin lower-bin)) :initial-element 1 :element-type 'bit)))
                (setf (subseq tpsd lower-bin (1+ upper-bin)) ones)
                tpsd))
         (expected-output (list 2 3 4 5))
         (my-output (br::thresh-psd-to-chan-idx psd current-channelization br::+radio-sample-rate+)))
    (is equalp (sort expected-output #'<) (sort my-output #'<))))

(define-test (interf-tensor main-test)
  "The main unit test of the `get-interference-tensor' function."
  ;; set-up: two collaborator teams, 3 nodes in my network, 25MHz
  ;; environment. node 1 measures both teams' power. node 2 measures only team
  ;; 1, node 3 measures only team 2.
  (let* ((br::*channelization* br::+debug-channelization-table+)
         (scen-rfb (floor 25e6))
         (scen-cfreq (floor 1e9))
         (env (make-instance 'br::environment
                             :rf-bandwidth scen-rfb
                             :center-freq scen-cfreq))
         (teams (list (make-instance 'br::network :id 1)
                      (make-instance 'br::network :id 2)))
         (fudge 1632)
         (cc (gethash scen-rfb br::+debug-channelization-table+))
         (assumed-bw (gethash (car cc) br::+debug-bandwidth-table+))
         (halfbw (/ assumed-bw 2)))
    (flet ((lower-absolute (chan-idx)
             (+ scen-cfreq (- (elt (cdr cc) chan-idx) halfbw)))
           (upper-absolute (chan-idx)
             (+ scen-cfreq (+ (elt (cdr cc) chan-idx) halfbw)))
           (make-su (band tid)
             (make-instance 'br::spectrum-usage
                            :users (list (make-instance 'br::spectrum-user
                                                        :network-id tid))
                            :band (make-instance 'br::frequency-band
                                                 :start (car band)
                                                 :stop (cdr band)))))
      (let* ((team1-bands (list (cons (+ (lower-absolute 2) fudge)
                                      (- (upper-absolute 4) fudge))
                                (cons (+ (lower-absolute 20) fudge)
                                      (+ (upper-absolute 21) fudge))))
             (team2-bands (list (cons (- (lower-absolute 8) fudge)
                                      (+ (upper-absolute 8) fudge))
                                (cons (- (lower-absolute 14) fudge)
                                      (+ (upper-absolute 17) fudge))))
             (my-chan-idx '(10 12 18))
             (declspec (append (mapcar #'(lambda (band) (make-su band 1)) team1-bands)
                               (mapcar #'(lambda (band) (make-su band 2)) team2-bands))))
        (flet ((make-psd (bands)
                 (let* ((npsd 1024)
                        (tpsd (make-array npsd :initial-element 0 :element-type 'bit)))
                   (loop for band in bands do
                        (let* ((lower (- (car band) scen-cfreq))
                               (upper (- (cdr band) scen-cfreq))
                               (lower-bin (br::freq-to-bin lower br::+radio-sample-rate+ npsd))
                               (upper-bin (br::freq-to-bin upper br::+radio-sample-rate+ npsd))
                               (ones (make-array (1+ (- upper-bin lower-bin)) :initial-element 1 :element-type 'bit)))
                          (setf (subseq tpsd lower-bin (1+ upper-bin)) ones)))
                   tpsd)))
          (let* ((my-bands (mapcar #'(lambda (idx)
                                       (cons (lower-absolute idx)
                                             (upper-absolute idx)))
                                   my-chan-idx))
                 (my-nodes (list (make-instance 'br::internal-node
                                                :id 1
                                                :thresh-psd (make-psd (append team1-bands
                                                                              team2-bands
                                                                              my-bands)))
                                 (make-instance 'br::internal-node
                                                :id 2
                                                :thresh-psd (make-psd (append team1-bands
                                                                              my-bands)))
                                 (make-instance 'br::internal-node
                                                :id 3
                                                :thresh-psd (make-psd (append team2-bands
                                                                              my-bands)))))
                 (fake-input (make-instance 'br::decision-engine-input
                                            :env env
                                            :collaborators teams
                                            :declared-spectrum declspec
                                            :nodes my-nodes))
                 (expected-output (let ((ht1 (make-hash-table))
                                        (nchan (length (cdr cc))))
                                    (setf (gethash 1 ht1) ;; team 1
                                          (let ((ht2 (make-hash-table)))
                                            (setf (gethash 1 ht2) ;; node 1
                                                  (br::chan-idx-list-to-binary '(2 3 4 20 21) nchan))
                                            (setf (gethash 2 ht2) ;; node 2
                                                  (br::chan-idx-list-to-binary '(2 3 4 20 21) nchan))
                                            (setf (gethash 3 ht2) ;; node 3
                                                  (br::chan-idx-list-to-binary '() nchan))
                                            ht2))
                                    (setf (gethash 2 ht1) ;; team 2
                                          (let ((ht2 (make-hash-table)))
                                            (setf (gethash 1 ht2) ;; node 1
                                                  (br::chan-idx-list-to-binary '(8 14 15 16 17) nchan))
                                            (setf (gethash 2 ht2) ;; node 2
                                                  (br::chan-idx-list-to-binary '() nchan))
                                            (setf (gethash 3 ht2) ;; node 3
                                                  (br::chan-idx-list-to-binary '(8 14 15 16 17) nchan))
                                            ht2))
                                    ht1))
                 (my-output))
            (setf my-output (br::get-interference-tensor fake-input))
            (when *verbose-output*
              (br::print-nested-hash-table my-output)
              (br::print-nested-hash-table expected-output))
            ;; FIXME: This equalp is correct even when it shouldn't be. rely on manual inspection for now.
            (is equalp expected-output my-output)))))))

;; lowest-sp-trend-least-collisions tests -- TODO

(define-test lowest-sp-trend-least-collisions
   :parent bam-radio)

(define-test (lowest-sp-trend-least-collisions get-collision-count)
  "Two nodes, two collaborators. We test whether `get-collision-count' returns
   the correct counts."
  (flet ((empty-collision-count (nchan)
           (mapcar #'(lambda (idx) (cons idx 0)) (alexandria:iota nchan))))
    (let* ((nchan 17)
           (interf-tensor
            (let ((ht1 (make-hash-table)))
              (setf (gethash 1 ht1) ;; team 1
                    (let ((ht2 (make-hash-table)))
                      (setf (gethash 56 ht2) ;; node 56
                            (br::chan-idx-list-to-binary '(0 1 2 6 7 8 9) nchan))
                      (setf (gethash 33 ht2) ;; node 33
                            (br::chan-idx-list-to-binary '(0 1 2 6 7 8 9) nchan))
                      ht2))
              (setf (gethash 3 ht1) ;; team 2
                    (let ((ht2 (make-hash-table)))
                      (setf (gethash 56 ht2) ;; node 56
                            (br::chan-idx-list-to-binary '(6 7 13 14 15) nchan))
                      (setf (gethash 33 ht2) ;; node 33
                            (br::chan-idx-list-to-binary '(6 7) nchan))
                      ht2))
              ht1))
           (expected-output-56 (let ((c (empty-collision-count nchan)))
                                 (setf (cdr (elt c 0)) 1)
                                 (setf (cdr (elt c 1)) 1)
                                 (setf (cdr (elt c 2)) 1)
                                 (setf (cdr (elt c 6)) 2)
                                 (setf (cdr (elt c 7)) 2)
                                 (setf (cdr (elt c 8)) 1)
                                 (setf (cdr (elt c 9)) 1)
                                 (setf (cdr (elt c 13)) 1)
                                 (setf (cdr (elt c 14)) 1)
                                 (setf (cdr (elt c 15)) 1)
                                 c))
           (expected-output-33 (let ((c (empty-collision-count nchan)))
                                 (setf (cdr (elt c 0)) 1)
                                 (setf (cdr (elt c 1)) 1)
                                 (setf (cdr (elt c 2)) 1)
                                 (setf (cdr (elt c 6)) 2)
                                 (setf (cdr (elt c 7)) 2)
                                 (setf (cdr (elt c 8)) 1)
                                 (setf (cdr (elt c 9)) 1)
                                 c))
           (my-output-33)
           (my-output-56))
      (setf my-output-33 (br::get-collision-count interf-tensor 33 nchan))
      (setf my-output-56 (br::get-collision-count interf-tensor 56 nchan))
      (when *verbose-output* ; debug output
        (format t "interf-tensor:~%")
        (br::print-nested-hash-table interf-tensor)
        (format t "ex: ~a~%my: ~a~%" expected-output-33 my-output-33)
        (format t "ex: ~a~%my: ~a~%" expected-output-56 my-output-56))
      (is equalp expected-output-33 my-output-33)
      (is equalp expected-output-56 my-output-56))))

(define-test (lowest-sp-trend-least-collisions rank-by-active-mandate-metric)
  "Two transmitters, two mandates per transmitter. we rank by the sum of the
  values of the scalar performance field."
  (let* ((mandates (list (make-instance 'br::mandate
                                        :id 10
                                        :tx 33
                                        :rx 91
                                        :active t
                                        :performance (make-instance 'br::mandate-performance
                                                                    :scalar-performance 10))
                         (make-instance 'br::mandate
                                        :id 20
                                        :tx 33
                                        :rx 91
                                        :active t
                                        :performance (make-instance 'br::mandate-performance
                                                                    :scalar-performance 20))
                         (make-instance 'br::mandate
                                        :id 5
                                        :tx 56
                                        :rx 91
                                        :active t
                                        :performance (make-instance 'br::mandate-performance
                                                                    :scalar-performance 5))
                         (make-instance 'br::mandate
                                        :id 30
                                        :tx 56
                                        :rx 91
                                        :active t
                                        :performance (make-instance 'br::mandate-performance
                                                                    :scalar-performance 30))))
         (nodes (list (make-instance 'br::internal-node :id 33)
                      (make-instance 'br::internal-node :id 56)))
         (fake-input (make-instance 'br::decision-engine-input
                                    :nodes nodes
                                    :mandates mandates))
         (expected-output (list (cons 33 30) (cons 56 35)))
         (my-output))
    (flet ((compute-sum-sp (active-mandates)
             (let ((sum 0))
               (loop for mandate in active-mandates do
                    (setf sum (+ sum (br::scalar-performance (br::performance mandate)))))
               sum)))
      (setf my-output (br::rank-by-active-mandate-metric fake-input #'compute-sum-sp))
      (when *verbose-output* ; debug-output
        (format t "ex: ~A~%my: ~A~%" expected-output my-output))
      (is equalp expected-output my-output))))

(define-test (lowest-sp-trend-least-collisions compute-sp-trend)
  "Two mandates, compute the trend. One of them does not show up in the
  previous inputs."
  (let* ((mandates (list (make-instance 'br::mandate
                                        :id 10
                                        :tx 33
                                        :active t
                                        :performance (make-instance 'br::mandate-performance
                                                                    :scalar-performance 10))
                         (make-instance 'br::mandate
                                        :id 20
                                        :tx 33
                                        :active t
                                        :performance (make-instance 'br::mandate-performance
                                                                    :scalar-performance 20))))
         (prev-mandates (list (make-instance 'br::mandate
                                        :id 10
                                        :tx 33
                                        :active t
                                        :performance (make-instance 'br::mandate-performance
                                                                    :scalar-performance 11))))
         (br::*prev-inputs* (list (make-instance 'br::decision-engine-input :mandates prev-mandates)))
         (expected-output 0.5)
         (my-output))
    (setf my-output (br::compute-sp-trend mandates))
    (when *verbose-output*
      (format t "ex: ~A~%my: ~A~%" expected-output my-output))
    (is = expected-output my-output)))

(define-test (lowest-sp-trend-least-collisions lstlc-main)
  :depends-on (compute-sp-trend rank-by-active-mandate-metric get-collision-count)
  (let* ((env (make-instance 'br::environment
                             :rf-bandwidth (floor 8e6)))
         (nchan (length (cdr (gethash (br::rf-bandwidth env) br::+debug-channelization-table+))))
         ;; nodes and their tx-assignments
         (nodes (list (make-instance 'br::internal-node
                                     :id 33
                                     :tx-assignment (make-instance 'br::transmit-assignment
                                                                   :bw-idx 0
                                                                   :chan-idx 4))
                      (make-instance 'br::internal-node
                                     :id 56
                                     :tx-assignment (make-instance 'br::transmit-assignment
                                                                   :bw-idx 0
                                                                   :chan-idx 8))))
         ;; we are being interfered with on all bands, but there is one unique
         ;; minimum band
         (interf-tensor
          (let ((ht1 (make-hash-table)))
            (setf (gethash 1 ht1) ;; team 1
                  (let ((ht2 (make-hash-table)))
                    (setf (gethash 56 ht2) ;; node 56
                          (br::chan-idx-list-to-binary (alexandria:iota nchan) nchan))
                    (setf (gethash 33 ht2) ;; node 33
                          (br::chan-idx-list-to-binary (alexandria:iota nchan) nchan))
                    ht2))
            (setf (gethash 3 ht1) ;; team 2 -- does not interfere on channel 16
                  (let ((ht2 (make-hash-table)))
                    (setf (gethash 56 ht2) ;; node 56
                          (br::chan-idx-list-to-binary (alexandria:iota (1- nchan)) nchan))
                    (setf (gethash 33 ht2) ;; node 33
                          (br::chan-idx-list-to-binary (alexandria:iota (1- nchan)) nchan))
                    ht2))
            ht1))
         ;; doctored mandates such that node 33 is selected using the metric
         (mandates (list (make-instance 'br::mandate
                                        :id 10
                                        :tx 33
                                        :rx 91
                                        :active t
                                        :performance (make-instance 'br::mandate-performance
                                                                    :scalar-performance 0.6))
                         (make-instance 'br::mandate
                                        :id 20
                                        :tx 33
                                        :rx 91
                                        :active t
                                        :performance (make-instance 'br::mandate-performance
                                                                    :scalar-performance 0.5))
                         (make-instance 'br::mandate
                                        :id 5
                                        :tx 56
                                        :rx 91
                                        :active t
                                        :performance (make-instance 'br::mandate-performance
                                                                    :scalar-performance 0.7))
                         (make-instance 'br::mandate
                                        :id 30
                                        :tx 56
                                        :rx 91
                                        :active t
                                        :performance (make-instance 'br::mandate-performance
                                                                    :scalar-performance 0.8))))
         (prev-mandates (list (make-instance 'br::mandate
                                             :id 10
                                             :tx 33
                                             :rx 91
                                             :active t
                                             :performance (make-instance 'br::mandate-performance
                                                                         :scalar-performance 0.8))
                              (make-instance 'br::mandate
                                             :id 20
                                             :tx 33
                                             :rx 91
                                             :active t
                                             :performance (make-instance 'br::mandate-performance
                                                                         :scalar-performance 0.4))
                              (make-instance 'br::mandate
                                             :id 5
                                             :tx 56
                                             :rx 91
                                             :active t
                                             :performance (make-instance 'br::mandate-performance
                                                                         :scalar-performance 0.6))
                              (make-instance 'br::mandate
                                             :id 30
                                             :tx 56
                                             :rx 91
                                             :active t
                                             :performance (make-instance 'br::mandate-performance
                                                                         :scalar-performance 0.7))))
         (br::*prev-inputs* (list (make-instance 'br::decision-engine-input
                                                 :mandates prev-mandates)))
         ;; put input together and extract the tx assignment
         (fake-input (make-instance 'br::decision-engine-input
                                    :env env
                                    :nodes nodes
                                    :mandates mandates))
         (assgn (br::extract-tx-assignments fake-input)))
    ;; run the function
    (is = 4 (br::chan-idx (gethash 33 (br::assignment-map assgn))))
    (br::lowest-sp-trend-least-collisions assgn fake-input interf-tensor nil t)
    ;; check that assgn changed the channel of node 33 to the minimum channel
    (when *verbose-output*
      (br::print-nested-hash-table (br::assignment-map assgn)))
    (is = (1- nchan) (br::chan-idx (gethash 33 (br::assignment-map assgn))))
    (is eq t (br::channel-updated? assgn))))

(define-test (lowest-sp-trend-least-collisions rel-points-weight)
  "I am node 3 and I am transmitting to nodes 5, 1, and 4."
  (let* ((mandates (list (make-instance 'br::mandate :id 0 :tx 3 :rx 5 :point-value 25)
                         (make-instance 'br::mandate :id 1 :tx 3 :rx 5 :point-value 25)
                         (make-instance 'br::mandate :id 2 :tx 3 :rx 1 :point-value 10)
                         (make-instance 'br::mandate :id 3 :tx 3 :rx 4 :point-value 40)))
         (expected-output '((1 . 0.1) (4 . 0.4) (5 . 0.5)))
         (my-output))
    (setf my-output (sort (br::compute-rel-points-weight mandates)
                          #'(lambda (a b) (< (car a) (car b)))))
    (when *verbose-output*
      (format t "ex: ~A~%my: ~A~%" expected-output my-output))
    (is equalp expected-output my-output)))

(define-test (lowest-sp-trend-least-collisions rx-collision-metric)
  :depends-on (rel-points-weight)
  "Same flow set-up as above. three collaborator teams. expected result
  determined by hand + MATLAB."
  (defun float-nearly-eq (a b eps)
    "this is a bad float comparison."
    (< (abs (- a b)) eps))
  (let* ((mandates (list (make-instance 'br::mandate :active t :id 0 :tx 3 :rx 5 :point-value 25)
                         (make-instance 'br::mandate :active t :id 1 :tx 3 :rx 5 :point-value 25)
                         (make-instance 'br::mandate :active t :id 2 :tx 3 :rx 1 :point-value 10)
                         (make-instance 'br::mandate :active t :id 3 :tx 3 :rx 4 :point-value 40)))
         (env (make-instance 'br::environment :rf-bandwidth (floor 8e6)))
         (fake-input (make-instance 'br::decision-engine-input
                                    :env env
                                    :mandates mandates))
         (nchan (length (br::channelization-cfreqs fake-input)))
         (interf-tensor (let ((ht1 (make-hash-table)))
                          (setf (gethash 1 ht1) ;; team 1
                                (let ((ht2 (make-hash-table)))
                                  (setf (gethash 5 ht2) ;; node 5
                                        (br::chan-idx-list-to-binary '(0 1 2 15) nchan))
                                  (setf (gethash 1 ht2) ;; node 1
                                        (br::chan-idx-list-to-binary '(0 1 2 15) nchan))
                                  (setf (gethash 4 ht2) ;; node 4
                                        (br::chan-idx-list-to-binary '(0 1 2 15) nchan))
                                  ht2))
                          (setf (gethash 2 ht1) ;; team 2
                                (let ((ht2 (make-hash-table)))
                                  (setf (gethash 5 ht2) ;; node 5
                                        (br::chan-idx-list-to-binary '(5 6 10 11 15 16) nchan))
                                  (setf (gethash 1 ht2) ;; node 1
                                        (br::chan-idx-list-to-binary '() nchan))
                                  (setf (gethash 4 ht2) ;; node 4
                                        (br::chan-idx-list-to-binary '(5 6 10 11 15 16) nchan))
                                  ht2))
                          (setf (gethash 3 ht1) ;; team 3
                                (let ((ht2 (make-hash-table)))
                                  (setf (gethash 5 ht2) ;; node 5
                                        (br::chan-idx-list-to-binary '() nchan))
                                  (setf (gethash 1 ht2) ;; node 1
                                        (br::chan-idx-list-to-binary '() nchan))
                                  (setf (gethash 4 ht2) ;; node 4
                                        (br::chan-idx-list-to-binary '(0 1 2 9 10 11) nchan))
                                  ht2))
                          ht1))
         (expected-output (mapcar #'(lambda (m idx) (cons idx m))
                                  '(1.4 1.4 1.4 0.0 0.0 0.9 0.9 0.0 0.0 0.4 1.3 1.3 0 0.0 0.0 1.9 0.9)
                                  (alexandria:iota nchan)))
         (my-output))
    (setf my-output (br::get-rx-collisions-rel-point-weighted interf-tensor 3 fake-input))
    (when *verbose-output*
      (format t "ex: ~A~%my: ~A~%" expected-output my-output))
    (is eq t (every #'(lambda (a b)
                        (and (= (car a) (car b))
                             (float-nearly-eq (cdr a) (cdr b) 0.01)))
                    expected-output my-output))))


;;-----------------------------------------------------------------------------
;; environment updates
;;-----------------------------------------------------------------------------

(define-test process-env
  :parent bam-radio)

(define-test (process-env prev-env-no-change)
  "A previous environment exists and its rf_bandwidth is the same as the new
  environment. `process-env-update' should return NIL."
  ;; set-up
  (let* ((br::*prev-inputs* `(,(make-instance 'br::decision-engine-input
                                              :env (make-instance 'br::environment
                                                                  :rf-bandwidth (floor 40e6))
                                              :nodes nil)))
         (fake-input (make-instance 'br::decision-engine-input
                                    :env (make-instance 'br::environment
                                                        :rf-bandwidth (floor 40e6))
                                    :nodes `(,(make-instance 'br::internal-node
                                                             :id 0
                                                             :est-duty-cycle 0.5
                                                             ;; works for 40 MHz, but not 20 MHz
                                                             :tx-assignment (make-instance 'br::transmit-assignment
                                                                                           :bw-idx 4
                                                                                           :chan-idx 34)))))
         (assgn (br::extract-tx-assignments fake-input)))
    ;; run the function
    (br::process-env-update assgn fake-input)
    ;; bw-idx and chan-idx should not have changed
    (let* ((expected-bw-idx (br::bw-idx (br::tx-assignment (car (br::nodes fake-input)))))
           (expected-chan-idx (br::chan-idx (br::tx-assignment (car (br::nodes fake-input)))))
           (my-bw-idx (br::bw-idx (gethash (br::id (car (br::nodes fake-input)))
                                           (br::assignment-map assgn))))
           (my-chan-idx (br::chan-idx (gethash (br::id (car (br::nodes fake-input)))
                                               (br::assignment-map assgn)))))
      (is = expected-bw-idx my-bw-idx)
      (is = expected-chan-idx my-chan-idx))))

(define-test (process-env less-bandwidth)
  "rf_bandwidth shrinks, we should have new chan_idx within the range of the
  new channelization and all bw_idx should be less than or equal the new
  max-bw-idx."
  ;; set-up
  (let ((br::*channelization* br::+debug-channelization-table+)
        (br::*bandwidths* br::+debug-bandwidth-table+)
        (br::*prev-inputs* `(,(make-instance 'br::decision-engine-input
                                          :env (make-instance 'br::environment
                                                              :rf-bandwidth (floor 40e6))))))
    (let* ((fake-input (make-instance 'br::decision-engine-input
                                      :env (make-instance 'br::environment
                                                          :rf-bandwidth (floor 20e6))
                                      :nodes `(,(make-instance 'br::internal-node
                                                               :id 0
                                                               :est-duty-cycle 0.5
                                                               ;; works for 40 MHz, but not 20 MHz
                                                               :tx-assignment (make-instance 'br::transmit-assignment
                                                                                             :bw-idx 4
                                                                                             :chan-idx 34)))))
           (assgn (br::extract-tx-assignments fake-input)))
      ;; run the function
      (br::process-env-update assgn fake-input)
      ;; check output
      (let* ((expected-channelization (gethash (br::rf-bandwidth (br::env fake-input))
                                               br::*channelization*))
             (expected-max-bw-idx (car expected-channelization))
             (expected-max-chan-idx (1- (length (cdr expected-channelization))))
             (my-bw-idx (br::bw-idx (gethash (br::id (car (br::nodes fake-input)))
                                             (br::assignment-map assgn))))
             (my-chan-idx (br::chan-idx (gethash (br::id (car (br::nodes fake-input)))
                                                 (br::assignment-map assgn))))
             (my-chan-changed (br::channel-updated? assgn))
             (my-bw-changed (br::bandwidth-updated? assgn)))
        (is <= expected-max-bw-idx my-bw-idx)
        (is <= expected-max-chan-idx my-chan-idx)
        (is eq t my-chan-changed)
        (is eq t my-bw-changed)))))

(define-test compute-cil-overlaps
  :parent bam-radio
  "Test the funciton `compute-cil-overlaps'."
  ;; use the same set-up from the `su2iv' test, see whether output is
  ;; correct. Two SRNs, one is being interfered with and the other is not.
  (let* ((rfb 8000000)
         (current-channelization (gethash rfb br::+debug-channelization-table+))
         (lower-upper (list '(-2861647 . -2185412)
                            '(-2085176 . -1797176)
                            '(244235 . 920471)
                            '(1797176 . 2861647)))
         (target-channels (list '(1 2)
                                '(3)
                                '(9 10)
                                '(13 14 15)))
         (fo 1235) ; Hz if we want to push some stuff around
         (cfreq 1000000000) ; 1 GHz scenario center frequency
         ;; (1) a band starting slightly before 1 and ending inside of 2
         ;; (2) a band starting and ending in 3
         ;; (3) a band starting inside of 9 and ending slightly outside of 4
         ;; (4) a band starting before 13 and ending after 15
         (interfering-bands (mapcar #'(lambda (lu ff)
                                        (let ((lower (+ (car lu) (car ff)))
                                              (upper (+ (cdr lu) (cdr ff))))
                                          (make-instance 'br::frequency-band
                                                         :start (+ lower cfreq)
                                                         :stop (+ upper cfreq))))
                                    lower-upper
                                    (list (cons (- fo) (- fo))
                                          (cons fo (- fo))
                                          (cons fo fo)
                                          (cons (- fo) fo))))
         (nodes (list (make-instance 'br::internal-node
                                     :id 44
                                     :tx-assignment (make-instance 'br::transmit-assignment
                                                                   :chan-idx 9
                                                                   :bw-idx 0))
                      (make-instance 'br::internal-node
                                     :id 33
                                     :tx-assignment (make-instance 'br::transmit-assignment
                                                                   :chan-idx 4
                                                                   :bw-idx 0))))
         (su (mapcar #'(lambda (band id)
                         (make-instance 'br::spectrum-usage
                                        :users (list (make-instance 'br::spectrum-user
                                                                    :network-id id))
                                        :band band))
                     interfering-bands '(1 2 1 2)))
         (fake-input (make-instance 'br::decision-engine-input
                                    :env (make-instance 'br::environment
                                                        :rf-bandwidth rfb
                                                        :center-freq cfreq)
                                    :collaborators (list (make-instance 'br::network :id 1)
                                                         (make-instance 'br::network :id 2))
                                    :nodes nodes
                                    :declared-spectrum su))
         (expected-output (sort (list (cons 44 t) (cons 33 nil)) #'< :key #'car))
         (my-output))
    (setf my-output (sort (br::compute-cil-overlaps (br::extract-tx-assignments fake-input)
                                                    fake-input)
                          #'< :key #'car))
    (when *verbose-output*
      (format t "ex: ~a~%my:~a~%" expected-output my-output))
    (is equalp expected-output my-output)))

;;-----------------------------------------------------------------------------
;; objective identification
;;-----------------------------------------------------------------------------

(define-test objective-identification
  :parent bam-radio)

(define-test (objective-identification rank-scores-func)
  (let* ((fake-input
	  (make-instance 'br::decision-engine-input
			 :env (make-instance 'br::environment
					     :bonus-threshold 1)
			 :mandates (list (make-instance 'br::mandate
							:performance (make-instance 'br::mandate-performance
										    :scalar-performance 1.1)
							:point-value 2))
			 :collaborators (list (make-instance 'br::network
							     :id 10
							     :reported-score 1
							     :scoring-point-threshold 100)
					      (make-instance 'br::network
							     :id 15
							     :reported-score 10
							     :scoring-point-threshold 50)
					      (make-instance 'br::network
							     :id 30
							     :reported-score 80
							     :scoring-point-threshold 60))))
	 (team-scores (br::rank-scores fake-input))
	 (expected-output '((2130706433 . 2) (30 . 80/60) (15 . 10/50) (10 . 1/100))))
    (is equalp team-scores expected-output)
    ))

(define-test (objective-identification rank-scores-func-2)
  (let* ((fake-input
	        (make-instance 'br::decision-engine-input
			                   :env (make-instance 'br::environment
					                                   :bonus-threshold 1)
			                   :mandates (list (make-instance 'br::mandate
							                                          :performance (make-instance 'br::mandate-performance
										                                                                :scalar-performance 1.1)
							                                          :point-value 2))
			                   :collaborators (list (make-instance 'br::network
							                                               :id 10
							                                               :reported-score 1
							                                               :scoring-point-threshold 100)
					                                    (make-instance 'br::network
							                                               :id 15
							                                               :reported-score nil
							                                               :scoring-point-threshold 50)
					                                    (make-instance 'br::network
							                                               :id 30
							                                               :reported-score 80
							                                               :scoring-point-threshold 60))))
	       (team-scores (br::rank-scores fake-input))
	       (expected-output '((2130706433 . 2) (30 . 80/60) (10 . 1/100))))
    (is equalp team-scores expected-output)))


(define-test (objective-identification get-objective-func)
  ;; PROTECT-INCUMBENT
  (let* ((fake-input
          (make-instance 'br::decision-engine-input
                         :incumbent (make-instance 'br::passive-incumbent
                                                   :last-message (make-instance 'br::passive-incumbent-message
                                                                                :threshold-exceeded t
                                                                                :threshold -60
                                                                                :power -59))))
         (fake-scores '((2130706433 . 1.5) (10 . 1.3) (15 . 0.9) (30 . 0.2)))
         (objective (br::get-objective fake-input fake-scores)))
    (is equal objective 'br::HANDLE-INCUMBENT-VIOLATION))
  ;; INCREASE-BONUS-SCORE
  (let* ((fake-input
          (make-instance 'br::decision-engine-input
                         :incumbent nil))
         (fake-scores '((2130706433 . 1.5) (10 . 1.3) (15 . 1.1) (30 . 1)))
         (objective (br::get-objective fake-input fake-scores)))
    (is equal objective 'br::INCREASE-BONUS-SCORE))
  ;; PROTECT-INCUMBENT-AND-INCREASE-BONUS-SCORE
  (let* ((fake-input
          (make-instance 'br::decision-engine-input
                         :incumbent (make-instance 'br::passive-incumbent
                                                   :last-message (make-instance 'br::passive-incumbent-message
                                                                                :threshold-exceeded nil))))
         (fake-scores '((2130706433 . 1.5) (10 . 1.3) (15 . 1.1) (30 . 1)))
         (objective (br::get-objective fake-input fake-scores)))
    (is equal objective 'br::PROTECT-INCUMBENT-AND-INCREASE-BONUS-SCORE))
  ;; PROTECT-WORST
  (let* ((fake-input
          (make-instance 'br::decision-engine-input
                         :incumbent nil))
         (fake-scores '((2130706433 . 1.5) (10 . 1.3) (15 . 0.9) (30 . 0.2)))
         (objective (br::get-objective fake-input fake-scores)))
    (is equal objective 'br::PROTECT-WORST))
  ;; PROTECT-INCUMBENT-AND-PROTECT-WORST
  (let* ((fake-input
          (make-instance 'br::decision-engine-input
                         :incumbent (make-instance 'br::passive-incumbent
                                                   :last-message (make-instance 'br::passive-incumbent-message
                                                                                :threshold-exceeded nil))))
         (fake-scores '((2130706433 . 1.5) (10 . 1.3) (15 . 0.9) (30 . 0.2)))
         (objective (br::get-objective fake-input fake-scores)))
    (is equal objective 'br::PROTECT-INCUMBENT-AND-PROTECT-WORST))
  ;; INCREASE-SCORE
  (let* ((fake-input
          (make-instance 'br::decision-engine-input
                         :incumbent nil))
         (fake-scores '((10 . 1.1) (15 . 0.6) (30 . 0.4) (2130706433 . 0.2)))
         (objective (br::get-objective fake-input fake-scores)))
    (is equal objective 'br::INCREASE-SCORE))
  ;; PROTECT-INCUMBENT-AND-INCREASE-SCORE
  (let* ((fake-input
          (make-instance 'br::decision-engine-input
                         :incumbent (make-instance 'br::passive-incumbent
                                                   :last-message (make-instance 'br::passive-incumbent-message
                                                                                :threshold-exceeded nil))))
         (fake-scores '((10 . 1.1) (15 . 0.6) (30 . 0.4) (2130706433 . 0.2)))
         (objective (br::get-objective fake-input fake-scores)))
    (is equal objective 'br::PROTECT-INCUMBENT-AND-INCREASE-SCORE))
  ;; PROTECT-WORST
  (let* ((fake-input
          (make-instance 'br::decision-engine-input
                         :incumbent nil))
         (fake-scores '((10 . 1.1) (2130706433 . 0.9) (15 . 0.6) (30 . 0.4)))
         (objective (br::get-objective fake-input fake-scores)))
    (is equal objective 'br::PROTECT-WORST))
  ;; PROTECT-INCUMBENT-AND-PROTECT-WORST
  (let* ((fake-input
          (make-instance 'br::decision-engine-input
                         :incumbent (make-instance 'br::passive-incumbent
                                                   :last-message (make-instance 'br::passive-incumbent-message
                                                                                :threshold-exceeded nil))))
         (fake-scores '((10 . 1.1) (2130706433 . 0.9) (15 . 0.6) (30 . 0.4)))
         (objective (br::get-objective fake-input fake-scores)))
    (is equal objective 'br::PROTECT-INCUMBENT-AND-PROTECT-WORST)))

;;-----------------------------------------------------------------------------
;; incumbent protection
;;-----------------------------------------------------------------------------

(define-test incumbent-protection
    :parent bam-radio)

(define-test (incumbent-protection incumbent-protection1)
    (let* ((new-assignments (make-hash-table)))
      (setf (gethash 100 new-assignments) (make-instance 'br::transmit-assignment
							 :bw-idx 0
							 :chan-idx 13))
      (setf (gethash 101 new-assignments) (make-instance 'br::transmit-assignment
							 :bw-idx 0
							 :chan-idx 4))
      (let* ((fake-assgn (make-instance 'br::tx-assignment-update
					:assignment-map new-assignments))
	     (fake-input (make-instance 'br::decision-engine-input
					:env (make-instance 'br::environment
							    :rf-bandwidth 20000000
							    :center-freq 1000000000)
					:incumbent (make-instance 'br::passive-incumbent
								  :offset 2500000
								  :bandwidth 5000000))))
	(br::chan-alloc-protect-incumbent fake-assgn fake-input nil nil)
	(is eq (br::silent (gethash 100 (br::assignment-map fake-assgn))) t)
	(is eq (br::atten-updated? fake-assgn) t)
	(is eq (br::silent (gethash 101 (br::assignment-map fake-assgn))) nil))))

(define-test (incumbent-protection incumbent-protection2)
  (defun float-nearly-eq (a b eps)
    "this is a bad float comparison."
    (< (abs (- a b)) eps))

  ;; Passive incumbent: Threshold -60 dB, Power -61 dB
  (let* ((new-assignments (make-hash-table)))
    (setf (gethash 100 new-assignments) (make-instance 'br::transmit-assignment
                                                       :bw-idx 0
                                                       :chan-idx 13
                                                       :atten 10))
    (setf (gethash 101 new-assignments) (make-instance 'br::transmit-assignment
                                                       :bw-idx 0
                                                       :chan-idx 4
                                                       :atten 10))
    (let* ((fake-assgn (make-instance 'br::tx-assignment-update
                                      :assignment-map new-assignments))
           (fake-input (make-instance 'br::decision-engine-input
                                      :env (make-instance 'br::environment
                                                          :rf-bandwidth 20000000
                                                          :center-freq 1000000000)
                                      :incumbent (make-instance 'br::passive-incumbent
                                                                :offset 2500000
                                                                :bandwidth 5000000
                                                                :last-message (make-instance 'br::passive-incumbent-message
                                                                                             :threshold-exceeded nil
                                                                                             :threshold -60
                                                                                             :power -61)))))
      (br::chan-alloc-adjust-tx-power fake-assgn fake-input nil nil)
      (is eq (float-nearly-eq 14 (br::atten (gethash 100 (br::assignment-map fake-assgn))) 0.01) t)
      (is eq (float-nearly-eq 0 (br::atten (gethash 101 (br::assignment-map fake-assgn))) 0.01) t)
      (is eq (br::atten-updated? fake-assgn) t)))

  ;; Passive incumbent: Threshold -60 dB, Power -73 dB
  (let* ((new-assignments (make-hash-table)))
    (setf (gethash 100 new-assignments) (make-instance 'br::transmit-assignment
                                                       :bw-idx 0
                                                       :chan-idx 13
                                                       :atten 10))
    (let* ((fake-assgn (make-instance 'br::tx-assignment-update
                                      :assignment-map new-assignments))
           (fake-input (make-instance 'br::decision-engine-input
                                      :env (make-instance 'br::environment
                                                          :rf-bandwidth 20000000
                                                          :center-freq 1000000000)
                                      :incumbent (make-instance 'br::passive-incumbent
                                                                :offset 2500000
                                                                :bandwidth 5000000
                                                                :last-message (make-instance 'br::passive-incumbent-message
                                                                                             :threshold-exceeded nil
                                                                                             :threshold -60
                                                                                             :power -73)))))
      (br::chan-alloc-adjust-tx-power fake-assgn fake-input nil nil)
      (is eq (float-nearly-eq 5 (br::atten (gethash 100 (br::assignment-map fake-assgn))) 0.01) t)
      (is eq (br::atten-updated? fake-assgn) t)))

  ;; Active incumbent: Threshold -60 dB, Power -61 dB
  (let* ((new-assignments (make-hash-table)))
    (setf (gethash 100 new-assignments) (make-instance 'br::transmit-assignment
                                                       :bw-idx 0
                                                       :chan-idx 13
                                                       :atten 10))
    (setf (gethash 101 new-assignments) (make-instance 'br::transmit-assignment
                                                       :bw-idx 0
                                                       :chan-idx 4
                                                       :atten 10))
    (let* ((fake-assgn (make-instance 'br::tx-assignment-update
                                      :assignment-map new-assignments))
           (fake-input (make-instance 'br::decision-engine-input
                                      :env (make-instance 'br::environment
                                                          :rf-bandwidth 20000000
                                                          :center-freq 1000000000)
                                      :incumbent (make-instance 'br::active-incumbent
                                                                :offset 2500000
                                                                :bandwidth 5000000
                                                                :last-message (make-instance 'br::active-incumbent-message
                                                                                             :threshold-exceeded nil
                                                                                             :threshold -60
                                                                                             :inr -61)))))
      (br::chan-alloc-adjust-tx-power fake-assgn fake-input nil nil)
      (is eq (float-nearly-eq 14 (br::atten (gethash 100 (br::assignment-map fake-assgn))) 0.01) t)
      (is eq (float-nearly-eq 0 (br::atten (gethash 101 (br::assignment-map fake-assgn))) 0.01) t)
      (is eq (br::atten-updated? fake-assgn) t))))
