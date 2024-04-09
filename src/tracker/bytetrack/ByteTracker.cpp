#include "ByteTracker.h"

ByteTracker::ByteTracker(int frame_rate, int track_buffer, float track_thresh, float high_thresh, float match_thresh)
{
	this->track_thresh = track_thresh;
	this->high_thresh = high_thresh;
    this->match_thresh = match_thresh;

    this->frame_id = 0;
    this->max_time_lost = int(frame_rate / 30.0 * track_buffer);
}

ByteTracker::~ByteTracker()
= default;

void ByteTracker::update(const DETECTIONS &objects)
{

	////////////////// Step 1: Get detections //////////////////
	this->frame_id++;
	 std::vector<STrack> activated_stracks;
	 std::vector<STrack> refind_stracks;
	 std::vector<STrack> tmp_removed_stracks;
	 std::vector<STrack> tmp_lost_stracks;
	 std::vector<STrack> detections;
	 std::vector<STrack> detections_low;

	 std::vector<STrack> detections_cp;
	 std::vector<STrack> tracked_stracks_swap;
	 std::vector<STrack> resa, resb;

	 std::vector<STrack*> unconfirmed;
	 std::vector<STrack*> tmp_tracked_stracks;
	 std::vector<STrack*> strack_pool;
	 std::vector<STrack*> r_tracked_stracks;

	if (!objects.empty())
	{
		for (const auto & object : objects)
		{
			 std::vector<float> tlbr_;
			tlbr_.resize(4);
            tlbr_[0] = object.tlwh[0];
            tlbr_[1] = object.tlwh[1];
            tlbr_[2] = object.tlwh[0] +  object.tlwh[2];
            tlbr_[3] = object.tlwh[1] +  object.tlwh[3];

            float score = object.confidence;

			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score);
			if (score >= track_thresh)
			{
				detections.push_back(strack);
			}
			else
			{
				detections_low.push_back(strack);
			}
			
		}
	}

	// Add newly detected tracklets to tmp_tracked_stracks
	for (auto & tracked_strack : this->tracked_stracks)
	{
		if (!tracked_strack.is_activated)
			unconfirmed.push_back(&tracked_strack);
		else
			tmp_tracked_stracks.push_back(&tracked_strack);
	}

	////////////////// Step 2: First association, with IoU //////////////////
	strack_pool = joint_stracks(tmp_tracked_stracks, this->lost_stracks);
	STrack::multi_predict(strack_pool, this->kalman_filter);

	std::vector< std::vector<float> > dists;
	int dist_size = 0, dist_size_size = 0;
	dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

	 std::vector< std::vector<int> > matches;
	 std::vector<int> u_track, u_detection;
	linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);

	for (auto & matche : matches)
	{
		STrack *track = strack_pool[matche[0]];
		STrack *det = &detections[matche[1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	////////////////// Step 3: Second association, using low score dets //////////////////

	for (int i : u_detection)
	{
		detections_cp.push_back(detections[i]);
	}


	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());

	for (int i : u_track)
	{
		if (strack_pool[i]->state == TrackState::Tracked)
		{
			r_tracked_stracks.push_back(strack_pool[i]);
		}
	}

	dists.clear();
	dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

	for (auto & matche : matches)
	{
		STrack *track = r_tracked_stracks[matche[0]];
		STrack *det = &detections[matche[1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	for (int i : u_track)
	{
		STrack *track = r_tracked_stracks[i];
		if (track->state != TrackState::Lost)
		{
			track->mark_lost();
			tmp_lost_stracks.push_back(*track);
		}
	}

	// Deal with unconfirmed tracks, usually tracks with only one beginning frame
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	dists.clear();
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

	matches.clear();
	 std::vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

	for (auto & matche : matches)
	{
		unconfirmed[matche[0]]->update(detections[matche[1]], this->frame_id);
		activated_stracks.push_back(*unconfirmed[matche[0]]);
	}

	for (int i : u_unconfirmed)
	{
		STrack *track = unconfirmed[i];
		track->mark_removed();
		tmp_removed_stracks.push_back(*track);
	}

	////////////////// Step 4: Init new stracks //////////////////
	for (int i : u_detection)
	{
		STrack *track = &detections[i];
		if (track->score < this->high_thresh)
			continue;
		track->activate(this->kalman_filter, this->frame_id);
		activated_stracks.push_back(*track);
	}

	////////////////// Step 5: Update state //////////////////
	for (auto & lost_strack : this->lost_stracks)
	{
		if (this->frame_id - lost_strack.end_frame() > this->max_time_lost)
		{
			lost_strack.mark_removed();
			tmp_removed_stracks.push_back(lost_strack);
		}
	}

	for (auto & tracked_strack : this->tracked_stracks)
	{
		if (tracked_strack.state == TrackState::Tracked)
		{
			tracked_stracks_swap.push_back(tracked_strack);
		}
	}

	this->tracked_stracks.clear();
	this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

	//std::cout << activated_stracks.size() << std::endl;


	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
	for (const auto & lost_strack : tmp_lost_stracks)
	{
		this->lost_stracks.push_back(lost_strack);
	}

	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);

	for (const auto & removed_strack : tmp_removed_stracks)
	{
		this->removed_stracks.push_back(removed_strack);
	}
	

	remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

	this->tracked_stracks.clear();
	this->tracked_stracks.assign(resa.begin(), resa.end());
	this->lost_stracks.clear();
	this->lost_stracks.assign(resb.begin(), resb.end());

    this->output_stracks.clear();
	for (auto & tracked_strack : this->tracked_stracks)
	{
		if (tracked_strack.is_activated)
		{
			this->output_stracks.push_back(tracked_strack);
		}
	}
}
